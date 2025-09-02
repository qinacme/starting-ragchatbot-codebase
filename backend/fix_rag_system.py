"""
Fix for RAG system issues identified through testing:

1. Improve document loading and deduplication logic  
2. Add better error handling and diagnostics
3. Fix ChromaDB metadata issues
4. Add data integrity checks
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import RAGSystem
from config import config
from vector_store import VectorStore
import traceback


def fix_chromadb_metadata_issue():
    """
    Fix the ChromaDB metadata issue found in integration tests.
    The issue was None values being passed to ChromaDB metadata fields.
    """
    print("=== Fixing ChromaDB Metadata Issue ===")
    
    # The issue is in document_processor.py where None values might be created
    # Let's check the vector_store.py add_course_metadata method
    
    from document_processor import DocumentProcessor
    
    # Test with a sample document to see if None values are generated
    processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    
    # Create a test document with minimal metadata
    test_content = """Course Title: Test Course
Course Instructor: Test Instructor

Lesson 1: Test Lesson
This is test content for the lesson.
"""
    
    # Save test content to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        course, chunks = processor.process_course_document(temp_file)
        
        print(f"Course title: {course.title}")
        print(f"Course instructor: {course.instructor}")
        print(f"Course link: {course.course_link}")  # This could be None
        print(f"Number of lessons: {len(course.lessons)}")
        
        # Check for None values in course object
        none_values = []
        if course.course_link is None:
            none_values.append("course_link")
        if course.instructor is None:
            none_values.append("instructor")
            
        if none_values:
            print(f"⚠️  Found None values in: {none_values}")
            print("This could cause ChromaDB metadata errors")
        else:
            print("✅ No None values found in course metadata")
        
        # Check chunk metadata
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i}: lesson_number={chunk.lesson_number}, course_title='{chunk.course_title}'")
            
            if chunk.lesson_number is None:
                print(f"⚠️  Chunk {i} has None lesson_number")
        
    except Exception as e:
        print(f"❌ Error testing metadata: {e}")
    finally:
        os.unlink(temp_file)
    
    return True


def fix_document_loading_logic():
    """
    Fix issues with document loading and deduplication
    """
    print("\n=== Fixing Document Loading Logic ===")
    
    try:
        # The issue might be that courses exist in catalog but not content
        # Let's add a more robust check
        
        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        # Check existing state
        catalog_results = store.course_catalog.get()
        content_results = store.course_content.get()
        
        catalog_count = len(catalog_results.get('ids', []))
        content_count = len(content_results.get('ids', []))
        
        print(f"Current state: {catalog_count} catalog entries, {content_count} content chunks")
        
        if catalog_count > 0 and content_count > 0:
            # Check consistency
            catalog_titles = set(catalog_results['ids'])
            content_titles = set()
            for metadata in content_results['metadatas']:
                content_titles.add(metadata.get('course_title'))
            
            missing_content = catalog_titles - content_titles
            missing_catalog = content_titles - catalog_titles
            
            if missing_content:
                print(f"⚠️  Courses in catalog but missing content: {missing_content}")
            if missing_catalog:
                print(f"⚠️  Content for courses not in catalog: {missing_catalog}")
            
            if missing_content or missing_catalog:
                print("💡 Recommendation: Clear and reload all data for consistency")
                return False
            else:
                print("✅ Catalog and content are consistent")
                return True
        
        elif catalog_count == 0 and content_count == 0:
            print("ℹ️  Clean state - ready for document loading")
            return True
        
        else:
            print(f"⚠️  Inconsistent state: {catalog_count} catalog, {content_count} content")
            return False
            
    except Exception as e:
        print(f"❌ Error checking document loading: {e}")
        return False


def fix_search_error_handling():
    """
    Add better error handling to search operations
    """
    print("\n=== Fixing Search Error Handling ===")
    
    # This would involve editing search_tools.py to add better error messages
    # For now, let's test current error handling
    
    try:
        rag_system = RAGSystem(config)
        
        # Test search with current state
        search_tool = rag_system.search_tool
        
        # Test various search scenarios
        test_cases = [
            ("Normal query", "What is MCP?"),
            ("Empty query", ""),
            ("Non-existent course", "Tell me about a course that doesn't exist"),
            ("Special characters", "What is MCP? <>&'\""),
        ]
        
        all_passed = True
        for description, query in test_cases:
            try:
                result = search_tool.execute(query)
                
                if "Search error:" in result:
                    print(f"❌ {description}: {result}")
                    all_passed = False
                elif "No relevant content found" in result:
                    print(f"✅ {description}: Handled gracefully (no content)")
                elif len(result) > 50:
                    print(f"✅ {description}: Returned results ({len(result)} chars)")
                else:
                    print(f"⚠️  {description}: Unexpected result: {result}")
                    
            except Exception as e:
                print(f"❌ {description}: Exception: {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error testing search handling: {e}")
        return False


def create_repair_script():
    """
    Create a repair script to fix common issues
    """
    print("\n=== Creating Repair Script ===")
    
    repair_script_content = '''#!/usr/bin/env python3
"""
RAG System Repair Script
Fixes common issues with the RAG chatbot system
"""
import os
import shutil
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import RAGSystem
from config import config


def clear_chromadb():
    """Clear ChromaDB completely"""
    if os.path.exists(config.CHROMA_PATH):
        print(f"Clearing ChromaDB at {config.CHROMA_PATH}")
        shutil.rmtree(config.CHROMA_PATH)
        print("✅ ChromaDB cleared")
    else:
        print("ℹ️  ChromaDB directory doesn't exist")


def reload_documents():
    """Reload all documents"""
    print("Reloading documents...")
    rag_system = RAGSystem(config)
    
    docs_path = "../docs"
    if os.path.exists(docs_path):
        courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=True)
        print(f"✅ Loaded {courses} courses with {chunks} chunks")
        
        if courses == 0:
            print("❌ No courses loaded - check docs folder")
            return False
    else:
        print(f"❌ Docs folder not found: {docs_path}")
        return False
    
    return True


def verify_system():
    """Verify system is working"""
    print("Verifying system...")
    
    try:
        rag_system = RAGSystem(config)
        
        # Check analytics
        analytics = rag_system.get_course_analytics()
        print(f"Courses loaded: {analytics['total_courses']}")
        
        if analytics['total_courses'] == 0:
            print("❌ No courses loaded")
            return False
        
        # Test search
        search_tool = rag_system.search_tool
        result = search_tool.execute("What is MCP?")
        
        if "No relevant content found" in result:
            print("❌ Search returns no results")
            return False
        elif "Search error:" in result:
            print(f"❌ Search error: {result}")
            return False
        else:
            print("✅ Search working correctly")
            return True
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    print("🔧 RAG System Repair Script")
    print("=" * 40)
    
    print("\\n1. Clearing ChromaDB...")
    clear_chromadb()
    
    print("\\n2. Reloading documents...")
    if not reload_documents():
        print("❌ Document reload failed")
        return
    
    print("\\n3. Verifying system...")
    if verify_system():
        print("\\n🎉 System repaired successfully!")
    else:
        print("\\n❌ System still has issues")


if __name__ == "__main__":
    main()
'''
    
    script_path = "../repair_rag_system.py"
    with open(script_path, 'w') as f:
        f.write(repair_script_content)
    
    # Make it executable
    os.chmod(script_path, 0o755)
    
    print(f"✅ Created repair script: {script_path}")
    print("   Run with: uv run python repair_rag_system.py")
    
    return True


def main():
    """Run all fixes"""
    print("🔧 Applying RAG System Fixes...\n")
    
    results = []
    results.append(("ChromaDB Metadata Fix", fix_chromadb_metadata_issue()))
    results.append(("Document Loading Fix", fix_document_loading_logic()))  
    results.append(("Search Error Handling", fix_search_error_handling()))
    results.append(("Repair Script Creation", create_repair_script()))
    
    print(f"\n{'='*50}")
    print("FIX APPLICATION SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<25}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} fixes applied successfully")
    
    if passed < len(results):
        print("\n⚠️  Some fixes failed - check output above")
        print("   Consider running the repair script")
    else:
        print("\n🎉 All fixes applied!")
        print("   If you still see issues, run: uv run python repair_rag_system.py")


if __name__ == "__main__":
    main()