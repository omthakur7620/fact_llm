#!/usr/bin/env python3
"""
Interactive Fact Checker with detailed error handling
"""
import sys
import time
from datetime import datetime
import traceback
from app_config import config
from src.claim_extractor import ClaimExtractor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import EnhancedFactRetriever
from src.llm_client import GroqClient
from src.fact_checker import FactChecker

class InteractiveFactChecker:
    def __init__(self):
        self.system = None
        self.session_history = []
        
    def initialize_system(self):
        """Initialize the fact checking system"""
        print("🚀 Initializing Fact Checking System...")
        
        try:
            # Validate configuration
            config.validate_config()
            
            # Initialize components
            claim_extractor = ClaimExtractor()
            embedding_generator = EmbeddingGenerator(config.EMBEDDING_MODEL)
            vector_store = VectorStore(config.FAISS_INDEX_PATH, config.METADATA_PATH)
            
            # Load vector store
            print("📂 Loading vector database...")
            vector_store.load()
            
            # Use EnhancedFactRetriever for better results
            retriever = EnhancedFactRetriever(embedding_generator, vector_store)
            llm_client = GroqClient(config.GROQ_API_KEY, config.GROQ_MODEL)
            self.system = FactChecker(claim_extractor, retriever, llm_client)
            
            # Show system info
            stats = vector_store.get_stats()
            print("✅ System initialized successfully!")
            print(f"📊 Vector Database: {stats['total_entries']} documents loaded")
            print(f"🤖 Using Model: {config.GROQ_MODEL}")
            print(f"🎯 Similarity Threshold: {config.SIMILARITY_THRESHOLD}")
            
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def display_welcome(self):
        """Display welcome message and instructions"""
        print("\n" + "="*70)
        print("🔍 INTERACTIVE FACT CHECKER")
        print("="*70)
        print("Type any claim to check it against government press releases (2003)")
        print("\n💡 Example claims to try:")
        print("  • 'Indian Railways have heritage tourism programs'")
        print("  • 'Government announced free electricity for farmers'")
        print("  • 'SAIL is investing in steel plant upgrades'")
        print("  • 'Rural development funds were released to Assam'")
        print("\n🎯 Commands:")
        print("  • Type 'quit' or 'exit' to end session")
        print("  • Type 'history' to see previous checks")
        print("  • Type 'stats' to see session statistics")
        print("  • Type 'help' to show this message again")
        print("="*70)
    
    def check_claim_interactive(self, claim: str):
        """Check a single claim with detailed error handling"""
        print(f"\n🔍 Checking: \"{claim}\"")
        print("─" * 60)
        
        try:
            start_time = time.time()
            print("🔄 Step 1: Extracting claims...")
            result = self.system.check_claim(claim)
            processing_time = time.time() - start_time
            
            # Add to session history
            session_entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "claim": claim,
                "verdict": result['verdict'],
                "processing_time": round(processing_time, 2),
                "confidence": result.get('confidence', 'unknown')
            }
            self.session_history.append(session_entry)
            
            # Display results with better formatting
            self._display_result(result, processing_time)
            
            return result
            
        except Exception as e:
            print(f"❌ ERROR in check_claim_interactive: {e}")
            print("🔧 Full traceback:")
            traceback.print_exc()
            return None
    
    def _display_result(self, result, processing_time):
        """Display the fact-checking result in a user-friendly format"""
        
        if not result:
            print("❌ No result returned from fact-checking")
            return
        
        # Verdict with emoji
        verdict_emoji = {
            "TRUE": "✅",
            "LIKELY TRUE": "✅", 
            "FALSE": "❌",
            "LIKELY FALSE": "❌",
            "UNVERIFIABLE": "❓"
        }
        
        emoji = verdict_emoji.get(result['verdict'], "🔍")
        
        print(f"{emoji} VERDICT: {result['verdict']}")
        print(f"⏱️  Processed in: {processing_time:.2f}s")
        print(f"🎯 Confidence: {result.get('confidence', 'N/A').upper()}")
        
        print(f"\n📝 REASONING:")
        print(f"   {result['reasoning']}")
        
        if result.get('key_evidence'):
            print(f"\n🔎 KEY EVIDENCE:")
            for evidence in result['key_evidence'][:3]:
                print(f"   • {evidence}")
        
        if result.get('entities_found'):
            print(f"\n🏷️ ENTITIES IDENTIFIED:")
            entities_display = []
            for entity in result['entities_found'][:6]:
                entities_display.append(f"{entity['text']} ({entity['label']})")
            print(f"   {', '.join(entities_display)}")
        
        if result.get('retrieved_facts_count', 0) > 0:
            print(f"\n📚 DOCUMENTS ANALYZED: {result['retrieved_facts_count']}")
            
            # Show top matching sources
            if result.get('verification_details'):
                sources = set()
                for detail in result['verification_details']:
                    facts = detail.get('verification_result', {}).get('retrieved_facts', [])
                    for fact in facts[:2]:
                        sources.add(fact.get('source', 'Unknown'))
                
                if sources:
                    print(f"   📄 Sources: {', '.join(list(sources)[:3])}")
        
        print("─" * 60)
    
    def show_session_history(self):
        """Show history of checks in current session"""
        if not self.session_history:
            print("\n📝 No checks in session history yet.")
            return
        
        print(f"\n📋 SESSION HISTORY ({len(self.session_history)} checks)")
        print("─" * 60)
        
        for i, entry in enumerate(self.session_history[-10:], 1):
            verdict_emoji = "✅" if "TRUE" in entry['verdict'] else "❌" if "FALSE" in entry['verdict'] else "❓"
            print(f"{i:2d}. {verdict_emoji} [{entry['timestamp']}] {entry['claim'][:50]}...")
            print(f"     ⏱️ {entry['processing_time']}s | 🎯 {entry['confidence']}")
    
    def show_session_stats(self):
        """Show session statistics"""
        if not self.session_history:
            print("\n📊 No data available yet. Start checking some claims!")
            return
        
        total_checks = len(self.session_history)
        verdicts = {}
        total_time = 0
        
        for entry in self.session_history:
            verdicts[entry['verdict']] = verdicts.get(entry['verdict'], 0) + 1
            total_time += entry['processing_time']
        
        avg_time = total_time / total_checks
        
        print(f"\n📊 SESSION STATISTICS")
        print("─" * 60)
        print(f"📈 Total Checks: {total_checks}")
        print(f"⏱️  Average Time: {avg_time:.2f}s")
        print(f"🎯 Verdict Distribution:")
        
        for verdict, count in verdicts.items():
            percentage = (count / total_checks) * 100
            emoji = "✅" if "TRUE" in verdict else "❌" if "FALSE" in verdict else "❓"
            print(f"   {emoji} {verdict}: {count} ({percentage:.1f}%)")
    
    def run_interactive_mode(self):
        """Main interactive loop"""
        if not self.initialize_system():
            return
        
        self.display_welcome()
        
        while True:
            try:
                user_input = input("\n📝 Enter claim: ").strip()
                
                if not user_input:
                    continue
                    
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using the Fact Checker!")
                    if self.session_history:
                        self.show_session_stats()
                    break
                    
                elif user_input.lower() == 'history':
                    self.show_session_history()
                    continue
                    
                elif user_input.lower() == 'stats':
                    self.show_session_stats()
                    continue
                    
                elif user_input.lower() == 'help':
                    self.display_welcome()
                    continue
                
                # Process the claim
                result = self.check_claim_interactive(user_input)
                if result is None:
                    print("💡 The claim could not be processed. Please try a different one.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session ended by user.")
                if self.session_history:
                    self.show_session_stats()
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("🔧 Full traceback:")
                traceback.print_exc()
                print("💡 Please try again with a different claim.")

def main():
    """Main function for interactive mode"""
    checker = InteractiveFactChecker()
    checker.run_interactive_mode()

if __name__ == "__main__":
    main()