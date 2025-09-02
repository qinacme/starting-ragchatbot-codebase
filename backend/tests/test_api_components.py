"""
Test the API components without running the full FastAPI app
This isolates potential issues in the API layer
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from config import config
import traceback
import asyncio


async def test_api_query_logic():
    """Test the exact logic used in the /api/query endpoint"""
    print("=== Testing API Query Logic ===")
    
    try:
        # Initialize RAG system like the app does
        rag_system = RAGSystem(config)
        
        # Load documents like startup event does
        script_dir = os.path.dirname(os.path.abspath(__file__))
        docs_path = os.path.join(script_dir, "..", "..", "docs")
        
        if os.path.exists(docs_path):
            print("Loading documents like startup event...")
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            print(f"Loaded {courses} courses with {chunks} chunks")
            
            if courses == 0 and chunks == 0:
                print("❌ No documents loaded - this would cause 'query failed'")
                return False
        else:
            print("❌ Docs path doesn't exist - this would cause 'query failed'")
            return False
        
        # Test the exact query flow from the API endpoint
        test_queries = [
            {"query": "What is MCP?", "session_id": None},
            {"query": "Tell me about computer use", "session_id": "test-session"},
            {"query": "How does retrieval work?", "session_id": "test-session"},
        ]
        
        for i, request_data in enumerate(test_queries):
            print(f"\n--- API Test {i+1}: '{request_data['query']}' ---")
            
            try:
                # Simulate the exact logic from the API endpoint
                session_id = request_data.get('session_id')
                if not session_id:
                    session_id = rag_system.session_manager.create_session()
                    print(f"Created new session: {session_id}")
                
                # Process query using RAG system (same as API)
                answer, sources = rag_system.query(request_data['query'], session_id)
                
                print(f"✅ Query processed successfully")
                print(f"Answer length: {len(answer)}")
                print(f"Sources count: {len(sources)}")
                print(f"Answer preview: {answer[:150]}...")
                
                # Check for the specific error
                if "query failed" in answer.lower():
                    print("❌ Found 'query failed' in response!")
                    print(f"Full answer: {answer}")
                    return False
                    
                # Validate response structure (like API does)
                if not answer:
                    print("❌ Empty answer returned")
                    return False
                    
                if not isinstance(sources, list):
                    print(f"❌ Sources not a list: {type(sources)}")
                    return False
                
            except Exception as e:
                print(f"❌ Query processing failed: {e}")
                print("This would result in HTTP 500 error")
                traceback.print_exc()
                return False
        
        print("\n✅ All API query logic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ API logic test error: {e}")
        traceback.print_exc()
        return False


def test_session_manager():
    """Test session manager functionality used by API"""
    print("\n=== Testing Session Manager ===")
    
    try:
        rag_system = RAGSystem(config)
        
        # Test creating sessions
        session1 = rag_system.session_manager.create_session()
        print(f"✅ Created session: {session1}")
        
        # Test adding exchanges  
        rag_system.session_manager.add_exchange(session1, "Test query", "Test response")
        print("✅ Added exchange to session")
        
        # Test getting history
        history = rag_system.session_manager.get_conversation_history(session1)
        print(f"✅ Retrieved history: {len(history)} chars")
        
        # Test clearing session
        rag_system.session_manager.clear_session(session1)
        print("✅ Cleared session")
        
        return True
        
    except Exception as e:
        print(f"❌ Session manager error: {e}")
        return False


def test_error_conditions():
    """Test conditions that might cause 'query failed'"""
    print("\n=== Testing Error Conditions ===")
    
    try:
        rag_system = RAGSystem(config)
        
        # Test 1: Query with no documents loaded
        print("--- Test 1: No documents loaded ---")
        try:
            answer, sources = rag_system.query("Test query")
            print(f"Answer with no docs: {answer[:100]}...")
            
            if "no relevant content found" in answer.lower():
                print("✅ Handles no documents gracefully")
            else:
                print("⚠️  Unexpected response with no documents")
                
        except Exception as e:
            print(f"❌ Query with no docs failed: {e}")
            return False
        
        # Load documents for remaining tests
        script_dir = os.path.dirname(os.path.abspath(__file__))
        docs_path = os.path.join(script_dir, "..", "..", "docs")
        if os.path.exists(docs_path):
            rag_system.add_course_folder(docs_path)
        
        # Test 2: Invalid session ID
        print("--- Test 2: Invalid session ID ---")
        try:
            answer, sources = rag_system.query("Test query", "invalid-session-id")
            print("✅ Handles invalid session gracefully")
        except Exception as e:
            print(f"❌ Invalid session error: {e}")
            return False
        
        # Test 3: Extremely long query
        print("--- Test 3: Very long query ---")
        try:
            long_query = "What is MCP? " * 200  # Very long query
            answer, sources = rag_system.query(long_query)
            print(f"✅ Handled long query: {len(answer)} chars")
        except Exception as e:
            print(f"❌ Long query error: {e}")
            return False
        
        # Test 4: Special characters
        print("--- Test 4: Special characters ---")
        try:
            special_query = "What is MCP? 🤖 <script>alert('test')</script> & %20"
            answer, sources = rag_system.query(special_query)
            print("✅ Handled special characters")
        except Exception as e:
            print(f"❌ Special chars error: {e}")
            return False
        
        print("\n✅ All error condition tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error condition test failed: {e}")
        traceback.print_exc()
        return False


async def main():
    """Run API component tests"""
    print("🔍 Testing API Components (without full FastAPI)...\n")
    
    results = []
    results.append(("API Query Logic", await test_api_query_logic()))
    results.append(("Session Manager", test_session_manager()))
    results.append(("Error Conditions", test_error_conditions()))
    
    print(f"\n{'='*50}")
    print("API COMPONENTS TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"  
        print(f"{name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed < len(results):
        print("\n🔧 Issues found in API components")
    else:
        print("\n✅ API components working correctly!")


if __name__ == "__main__":
    asyncio.run(main())