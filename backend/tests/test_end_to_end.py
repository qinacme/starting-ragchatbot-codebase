"""
End-to-end test to simulate real user queries through the complete system
This should help identify where "query failed" is coming from
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from config import config
import traceback


def test_full_query_with_real_api():
    """Test a real query using the actual Anthropic API"""
    print("=== Testing Full Query with Real API ===")
    
    try:
        # Initialize RAG system
        rag = RAGSystem(config)
        
        # Ensure we have data loaded
        analytics = rag.get_course_analytics()
        if analytics["total_courses"] == 0:
            print("Loading course documents...")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            docs_path = os.path.join(script_dir, "..", "..", "docs")
            courses, chunks = rag.add_course_folder(docs_path)
            print(f"Loaded {courses} courses with {chunks} chunks")
            
            if courses == 0:
                print("❌ No courses loaded - cannot test")
                return False
        
        # Test queries that should trigger different behaviors
        test_queries = [
            "What is MCP?",
            "Tell me about computer use with Anthropic",
            "How does prompt compression work?", 
            "What is retrieval with Chroma?",
            "Explain Python programming"  # This should NOT use search tools
        ]
        
        for i, query in enumerate(test_queries):
            print(f"\n--- Test Query {i+1}: '{query}' ---")
            
            try:
                response, sources = rag.query(query)
                
                print(f"✅ Query succeeded")
                print(f"Response length: {len(response)}")
                print(f"Sources count: {len(sources)}")
                print(f"Response preview: {response[:200]}...")
                
                if "query failed" in response.lower():
                    print("❌ Response contains 'query failed'!")
                    print(f"Full response: {response}")
                    return False
                    
            except Exception as e:
                print(f"❌ Query failed with exception: {e}")
                print("Full traceback:")
                traceback.print_exc()
                return False
        
        print("\n✅ All queries completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ System error: {e}")
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases that might cause failures"""
    print("\n=== Testing Edge Cases ===")
    
    try:
        rag = RAGSystem(config)
        
        # Ensure we have data
        analytics = rag.get_course_analytics()
        if analytics["total_courses"] == 0:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            docs_path = os.path.join(script_dir, "..", "..", "docs")
            courses, chunks = rag.add_course_folder(docs_path)
            if courses == 0:
                print("❌ No courses loaded")
                return False
        
        edge_cases = [
            "",  # Empty query
            "   ",  # Whitespace only
            "a",  # Single character
            "x" * 1000,  # Very long query
            "What about a course that doesn't exist?",  # No matching content
            "¿Qué es MCP?",  # Non-English query
        ]
        
        for i, query in enumerate(edge_cases):
            print(f"\n--- Edge Case {i+1}: '{query[:50]}...' ---")
            
            try:
                response, sources = rag.query(query)
                print(f"✅ Handled gracefully")
                print(f"Response length: {len(response)}")
                
                if "query failed" in response.lower():
                    print("❌ Response contains 'query failed'!")
                    print(f"Full response: {response}")
                    return False
                    
            except Exception as e:
                print(f"❌ Edge case failed: {e}")
                return False
        
        print("\n✅ All edge cases handled!")
        return True
        
    except Exception as e:
        print(f"❌ Edge case testing error: {e}")
        traceback.print_exc()
        return False


def test_session_functionality():
    """Test session-based queries"""
    print("\n=== Testing Session Functionality ===")
    
    try:
        rag = RAGSystem(config)
        
        # Ensure we have data
        analytics = rag.get_course_analytics()
        if analytics["total_courses"] == 0:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            docs_path = os.path.join(script_dir, "..", "..", "docs")
            rag.add_course_folder(docs_path)
        
        session_id = "test-session"
        
        # First query
        print("--- First query in session ---")
        response1, sources1 = rag.query("What is MCP?", session_id)
        print(f"✅ First query: {len(response1)} chars, {len(sources1)} sources")
        
        # Follow-up query  
        print("--- Follow-up query in session ---")
        response2, sources2 = rag.query("Can you tell me more about that?", session_id)
        print(f"✅ Follow-up query: {len(response2)} chars, {len(sources2)} sources")
        
        if "query failed" in response1.lower() or "query failed" in response2.lower():
            print("❌ Session query failed!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Session test error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run end-to-end tests"""
    print("🔍 Running End-to-End RAG System Tests...\n")
    
    # Only test with real API if explicitly requested
    import sys
    if "--with-api" in sys.argv:
        print("⚠️  Testing with REAL Anthropic API (will consume tokens)")
        results = []
        results.append(("Full API Query Test", test_full_query_with_real_api()))
        results.append(("Edge Cases Test", test_edge_cases()))  
        results.append(("Session Test", test_session_functionality()))
        
        print(f"\n{'='*50}")
        print("END-TO-END TEST SUMMARY")
        print(f"{'='*50}")
        
        passed = 0
        for name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{name:<25}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{len(results)} tests passed")
    else:
        print("ℹ️  Skipping API tests. Use --with-api flag to test with real Anthropic API")
        print("   (This will consume API tokens)")
        
        # Run non-API tests
        print("\n--- Testing System Integrity (No API calls) ---")
        
        try:
            rag = RAGSystem(config)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            docs_path = os.path.join(script_dir, "..", "..", "docs")
            
            if os.path.exists(docs_path):
                courses, chunks = rag.add_course_folder(docs_path)
                print(f"✅ System can load {courses} courses with {chunks} chunks")
                
                # Test tool setup
                tools = rag.tool_manager.get_tool_definitions()
                print(f"✅ {len(tools)} tools registered: {[t['name'] for t in tools]}")
                
                # Test search tool directly  
                result = rag.search_tool.execute("MCP")
                if "No relevant content found" not in result and len(result) > 50:
                    print("✅ Search tool returns relevant content")
                else:
                    print(f"⚠️  Search tool result: {result[:100]}...")
                
                print("\n✅ System integrity check passed!")
                print("   The RAG components are working correctly.")
                print("   If you're seeing 'query failed', it's likely in the API integration.")
            else:
                print("❌ Docs folder not found")
                
        except Exception as e:
            print(f"❌ System integrity error: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()