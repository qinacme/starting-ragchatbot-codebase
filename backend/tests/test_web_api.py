"""
Test the web API layer to isolate if the "query failed" issue is in FastAPI endpoints
"""
import sys
import os
import httpx
import asyncio
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from config import config
import traceback


async def test_api_endpoints():
    """Test API endpoints directly"""
    print("=== Testing Web API Endpoints ===")
    
    # Use httpx to test the FastAPI app directly
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        
        # Test 1: Get course stats
        print("\n--- Test 1: Get Course Stats ---")
        try:
            response = await client.get("/api/courses")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            
            if response.status_code != 200:
                print(f"❌ Course stats failed: {response.text}")
                return False
            else:
                print("✅ Course stats endpoint working")
                
        except Exception as e:
            print(f"❌ Course stats error: {e}")
            return False
        
        # Test 2: Simple query
        print("\n--- Test 2: Simple Query ---")
        try:
            query_data = {"query": "What is MCP?"}
            response = await client.post("/api/query", json=query_data)
            print(f"Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Query failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return False
            else:
                response_json = response.json()
                print(f"✅ Query succeeded")
                print(f"Answer length: {len(response_json.get('answer', ''))}")
                print(f"Sources count: {len(response_json.get('sources', []))}")
                print(f"Answer preview: {response_json.get('answer', '')[:200]}...")
                
                if "query failed" in response_json.get('answer', '').lower():
                    print("❌ API returned 'query failed'!")
                    print(f"Full answer: {response_json.get('answer')}")
                    return False
                    
        except Exception as e:
            print(f"❌ Query error: {e}")
            traceback.print_exc()
            return False
        
        # Test 3: Query with session
        print("\n--- Test 3: Query with Session ---")
        try:
            query_data = {"query": "Tell me about computer use", "session_id": "test-session"}
            response = await client.post("/api/query", json=query_data)
            print(f"Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Session query failed: {response.text}")
                return False
            else:
                response_json = response.json()
                print(f"✅ Session query succeeded")
                print(f"Session ID: {response_json.get('session_id')}")
                
                if "query failed" in response_json.get('answer', '').lower():
                    print("❌ Session API returned 'query failed'!")
                    return False
                    
        except Exception as e:
            print(f"❌ Session query error: {e}")
            return False
        
        # Test 4: Edge case queries
        print("\n--- Test 4: Edge Case Queries ---")
        edge_cases = [
            "",  # Empty
            "a",  # Single char
            "What about nonexistent content?",  # No matches
        ]
        
        for i, query_text in enumerate(edge_cases):
            try:
                query_data = {"query": query_text}
                response = await client.post("/api/query", json=query_data)
                
                if response.status_code != 200:
                    print(f"❌ Edge case {i+1} failed: {response.status_code}")
                    return False
                else:
                    response_json = response.json()
                    print(f"✅ Edge case {i+1} handled")
                    
                    if "query failed" in response_json.get('answer', '').lower():
                        print(f"❌ Edge case {i+1} returned 'query failed'!")
                        return False
                        
            except Exception as e:
                print(f"❌ Edge case {i+1} error: {e}")
                return False
        
        # Test 5: Clear session
        print("\n--- Test 5: Clear Session ---")
        try:
            clear_data = {"session_id": "test-session"}
            response = await client.post("/api/clear-session", json=clear_data)
            print(f"Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Clear session failed: {response.text}")
                return False
            else:
                response_json = response.json()
                print(f"✅ Clear session: {response_json.get('message')}")
                
        except Exception as e:
            print(f"❌ Clear session error: {e}")
            return False
        
        print("\n✅ All API endpoint tests passed!")
        return True


async def test_concurrent_requests():
    """Test multiple concurrent requests to check for race conditions"""
    print("\n=== Testing Concurrent Requests ===")
    
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        
        # Create multiple concurrent queries
        queries = [
            "What is MCP?",
            "Tell me about Anthropic",
            "How does Chroma work?",
            "Explain retrieval systems",
        ]
        
        tasks = []
        for i, query in enumerate(queries):
            query_data = {"query": query, "session_id": f"session-{i}"}
            task = client.post("/api/query", json=query_data)
            tasks.append(task)
        
        try:
            # Wait for all requests
            print("Sending 4 concurrent queries...")
            responses = await asyncio.gather(*tasks)
            
            success_count = 0
            for i, response in enumerate(responses):
                if response.status_code == 200:
                    response_json = response.json()
                    if "query failed" not in response_json.get('answer', '').lower():
                        success_count += 1
                        print(f"✅ Concurrent query {i+1} succeeded")
                    else:
                        print(f"❌ Concurrent query {i+1} returned 'query failed'")
                else:
                    print(f"❌ Concurrent query {i+1} failed: {response.status_code}")
            
            if success_count == len(queries):
                print("✅ All concurrent requests succeeded!")
                return True
            else:
                print(f"❌ Only {success_count}/{len(queries)} concurrent requests succeeded")
                return False
                
        except Exception as e:
            print(f"❌ Concurrent request error: {e}")
            traceback.print_exc()
            return False


def test_startup_behavior():
    """Test if startup behavior causes issues"""
    print("\n=== Testing Startup Behavior ===")
    
    try:
        # This simulates what happens when the app starts
        from rag_system import RAGSystem
        
        print("Creating new RAG system instance...")
        rag = RAGSystem(config)
        
        # Check if documents are loaded
        analytics = rag.get_course_analytics()
        print(f"Courses loaded: {analytics['total_courses']}")
        
        if analytics['total_courses'] == 0:
            print("No courses loaded - this could cause 'query failed'")
            
            # Try to load documents like the startup event does
            script_dir = os.path.dirname(os.path.abspath(__file__))
            docs_path = os.path.join(script_dir, "..", "..", "docs")
            
            if os.path.exists(docs_path):
                print("Loading documents...")
                courses, chunks = rag.add_course_folder(docs_path, clear_existing=False)
                print(f"Loaded {courses} courses, {chunks} chunks")
                
                if courses == 0:
                    print("❌ Document loading failed - this could cause 'query failed'!")
                    return False
            else:
                print("❌ Docs folder not found!")
                return False
        
        print("✅ Startup behavior looks correct")
        return True
        
    except Exception as e:
        print(f"❌ Startup test error: {e}")
        traceback.print_exc()
        return False


async def main():
    """Run web API tests"""
    print("🔍 Testing Web API Layer...\n")
    
    results = []
    
    # Test startup first  
    results.append(("Startup Behavior", test_startup_behavior()))
    
    # Test API endpoints
    results.append(("API Endpoints", await test_api_endpoints()))
    
    # Test concurrent requests
    results.append(("Concurrent Requests", await test_concurrent_requests()))
    
    print(f"\n{'='*50}")
    print("WEB API TEST SUMMARY") 
    print(f"{'='*50}")
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed < len(results):
        print("\n🔧 Issues found in web API layer - check detailed output above")
    else:
        print("\n🎉 Web API layer working correctly!")
        print("   If you're still seeing 'query failed', check:")
        print("   1. Network connectivity")
        print("   2. Frontend-backend communication")
        print("   3. Environment differences")


if __name__ == "__main__":
    asyncio.run(main())