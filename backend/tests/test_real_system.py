"""
Diagnostic script to test the real system with actual course documents
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from config import config
from document_processor import DocumentProcessor
from vector_store import VectorStore


def test_document_processor():
    """Test document processor with real course files"""
    print("=== Testing Document Processor ===")
    
    processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    # Get absolute path to docs folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(script_dir, "..", "..", "docs")
    
    if not os.path.exists(docs_path):
        print(f"❌ Documents folder {docs_path} not found")
        return False
        
    files = [f for f in os.listdir(docs_path) if f.endswith('.txt')]
    print(f"Found {len(files)} course files: {files}")
    
    for filename in files[:1]:  # Test just first file
        file_path = os.path.join(docs_path, filename)
        print(f"\n--- Processing {filename} ---")
        
        try:
            course, chunks = processor.process_course_document(file_path)
            print(f"✅ Course: {course.title}")
            print(f"✅ Instructor: {course.instructor}")
            print(f"✅ Lessons: {len(course.lessons)}")
            print(f"✅ Chunks: {len(chunks)}")
            
            # Check for None values in metadata that would cause ChromaDB issues
            for i, chunk in enumerate(chunks[:3]):  # Check first 3 chunks
                print(f"Chunk {i}: course_title={chunk.course_title}, lesson_number={chunk.lesson_number}")
                
            return True
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            return False


def test_vector_store():
    """Test vector store operations"""
    print("\n=== Testing Vector Store ===")
    
    try:
        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        print("✅ VectorStore initialized")
        
        # Test basic operations
        count = store.get_course_count()
        titles = store.get_existing_course_titles()
        print(f"✅ Current course count: {count}")
        print(f"✅ Existing courses: {titles}")
        
        # Test search with existing data
        if count > 0:
            print("\n--- Testing Search ---")
            results = store.search("introduction")
            if results.error:
                print(f"❌ Search error: {results.error}")
                return False
            else:
                print(f"✅ Search returned {len(results.documents)} results")
                return True
        else:
            print("⚠️  No courses loaded - cannot test search")
            return True
            
    except Exception as e:
        print(f"❌ VectorStore error: {e}")
        return False


def test_rag_system():
    """Test full RAG system"""
    print("\n=== Testing RAG System ===")
    
    try:
        rag = RAGSystem(config)
        print("✅ RAG System initialized")
        
        # Test analytics
        analytics = rag.get_course_analytics()
        print(f"✅ Analytics: {analytics}")
        
        # Test loading documents if none exist
        if analytics["total_courses"] == 0:
            print("\n--- Loading Documents ---")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            docs_path = os.path.join(script_dir, "..", "..", "docs")
            if os.path.exists(docs_path):
                courses, chunks = rag.add_course_folder(docs_path)
                print(f"✅ Loaded {courses} courses with {chunks} chunks")
                
                if courses == 0:
                    print("❌ No courses were loaded - this indicates a problem")
                    return False
            else:
                print("⚠️  No docs folder found")
                
        return True
        
    except Exception as e:
        print(f"❌ RAG System error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_tool():
    """Test search tool functionality"""
    print("\n=== Testing Search Tool ===")
    
    try:
        rag = RAGSystem(config)
        
        # Ensure we have data
        analytics = rag.get_course_analytics()
        if analytics["total_courses"] == 0:
            print("⚠️  No courses loaded - loading from docs...")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            docs_path = os.path.join(script_dir, "..", "..", "docs")
            if os.path.exists(docs_path):
                courses, chunks = rag.add_course_folder(docs_path)
                if courses == 0:
                    print("❌ Could not load courses")
                    return False
        
        # Test search tool directly
        search_tool = rag.search_tool
        
        print("--- Testing Basic Search ---")
        result = search_tool.execute("introduction")
        print(f"Search result length: {len(result)}")
        print(f"First 200 chars: {result[:200]}...")
        
        if "No relevant content found" in result:
            print("❌ Search returned no results")
            return False
        elif result.startswith("Search error:") or result.startswith("No course found"):
            print(f"❌ Search error: {result}")
            return False
        else:
            print("✅ Search returned results")
            print(f"✅ Sources tracked: {len(search_tool.last_sources)}")
            return True
            
    except Exception as e:
        print(f"❌ Search tool error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_key():
    """Test if Anthropic API key is configured"""
    print("\n=== Testing API Configuration ===")
    
    if not config.ANTHROPIC_API_KEY:
        print("❌ No Anthropic API key configured")
        return False
    elif config.ANTHROPIC_API_KEY.startswith("sk-ant-api"):
        print("✅ API key is configured")
        return True
    else:
        print("⚠️  API key format looks unusual")
        return True


def main():
    """Run all diagnostic tests"""
    print("🔍 Running RAG System Diagnostics...\n")
    
    results = []
    results.append(("API Key", test_api_key()))
    results.append(("Document Processor", test_document_processor()))
    results.append(("Vector Store", test_vector_store()))
    results.append(("RAG System", test_rag_system()))
    results.append(("Search Tool", test_search_tool()))
    
    print(f"\n{'='*50}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed < len(results):
        print("\n🔧 Issues detected - check the detailed output above")
    else:
        print("\n🎉 All diagnostics passed!")


if __name__ == "__main__":
    main()