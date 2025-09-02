"""
Diagnose ChromaDB state to confirm the data inconsistency issue
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore
from config import config
import traceback


def diagnose_chromadb():
    """Diagnose ChromaDB collections and data integrity"""
    print("🔍 Diagnosing ChromaDB State...\n")
    
    try:
        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        print("=== Course Catalog Collection ===")
        try:
            catalog_results = store.course_catalog.get()
            catalog_count = len(catalog_results.get('ids', []))
            print(f"Course catalog entries: {catalog_count}")
            
            if catalog_count > 0:
                print("Course titles in catalog:")
                for i, course_id in enumerate(catalog_results['ids'][:5]):  # Show first 5
                    metadata = catalog_results['metadatas'][i]
                    print(f"  {i+1}. {metadata.get('title', 'No title')}")
                if catalog_count > 5:
                    print(f"  ... and {catalog_count - 5} more")
            else:
                print("❌ No courses in catalog!")
                
        except Exception as e:
            print(f"❌ Error accessing course catalog: {e}")
        
        print("\n=== Course Content Collection ===")
        try:
            content_results = store.course_content.get()
            content_count = len(content_results.get('ids', []))
            print(f"Content chunks: {content_count}")
            
            if content_count > 0:
                print("Sample content chunks:")
                for i in range(min(3, content_count)):
                    metadata = content_results['metadatas'][i]
                    content = content_results['documents'][i]
                    print(f"  {i+1}. Course: {metadata.get('course_title', 'Unknown')}")
                    print(f"     Lesson: {metadata.get('lesson_number', 'Unknown')}")  
                    print(f"     Content: {content[:100]}...")
                print(f"  ... and {content_count - min(3, content_count)} more chunks")
            else:
                print("❌ No content chunks found!")
                
        except Exception as e:
            print(f"❌ Error accessing course content: {e}")
        
        print("\n=== Data Consistency Check ===")
        if catalog_count > 0 and content_count == 0:
            print("🚨 INCONSISTENCY DETECTED!")
            print("   Course catalog has entries but content collection is empty")
            print("   This is the root cause of 'query failed' errors")
            return "inconsistent"
        elif catalog_count == 0 and content_count == 0:
            print("ℹ️  Both collections are empty (clean state)")
            return "empty"
        elif catalog_count > 0 and content_count > 0:
            print("✅ Both collections have data (healthy state)")
            
            # Check if content matches catalog
            existing_titles = set(store.get_existing_course_titles())
            content_titles = set()
            for metadata in content_results['metadatas']:
                content_titles.add(metadata.get('course_title'))
            
            if existing_titles == content_titles:
                print("✅ Course titles match between catalog and content")
                return "healthy"
            else:
                print("⚠️  Course titles don't match between collections")
                print(f"   Catalog titles: {existing_titles}")
                print(f"   Content titles: {content_titles}")
                return "mismatched"
        else:
            print("❓ Unusual state: content without catalog")
            return "unusual"
            
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        traceback.print_exc()
        return "error"


def test_search_with_current_state():
    """Test search functionality with current ChromaDB state"""
    print("\n=== Testing Search with Current State ===")
    
    try:
        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        test_queries = [
            "MCP",
            "computer use", 
            "retrieval",
            "anthropic"
        ]
        
        for query in test_queries:
            print(f"\n--- Testing query: '{query}' ---")
            
            try:
                results = store.search(query)
                
                if results.error:
                    print(f"❌ Search error: {results.error}")
                elif results.is_empty():
                    print("⚠️  No results found")
                else:
                    print(f"✅ Found {len(results.documents)} results")
                    
            except Exception as e:
                print(f"❌ Search failed: {e}")
                
    except Exception as e:
        print(f"❌ Search test failed: {e}")


def recommend_fix(state):
    """Recommend fix based on diagnosed state"""
    print(f"\n{'='*50}")
    print("DIAGNOSIS & RECOMMENDATIONS")
    print(f"{'='*50}")
    
    if state == "inconsistent":
        print("🚨 ROOT CAUSE CONFIRMED: ChromaDB Data Inconsistency")
        print()
        print("ISSUE:")
        print("  - Course catalog has entries (deduplication sees 'existing' courses)")
        print("  - Content collection is empty (no searchable content)")
        print("  - Search operations fail → 'query failed' errors")
        print()
        print("RECOMMENDED FIX:")
        print("  1. Clear ChromaDB completely")
        print("  2. Reload all course documents")
        print("  3. Add integrity checks to prevent future issues")
        print()
        print("COMMANDS TO FIX:")
        print("  - Delete ./chroma_db directory")
        print("  - Restart the application")
        print("  - Or use the repair script (if created)")
        
    elif state == "empty":
        print("ℹ️  ChromaDB is empty - documents need to be loaded")
        print("This would cause 'query failed' until documents are loaded")
        
    elif state == "healthy":
        print("✅ ChromaDB appears healthy")
        print("If you're seeing 'query failed', check other components")
        
    elif state == "mismatched":
        print("⚠️  ChromaDB has mismatched data between collections")
        print("Recommend clearing and reloading for consistency")
        
    else:
        print("❓ Unable to diagnose - manual investigation needed")


def main():
    state = diagnose_chromadb()
    test_search_with_current_state()
    recommend_fix(state)


if __name__ == "__main__":
    main()