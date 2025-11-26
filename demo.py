#!/usr/bin/env python3
"""
Comprehensive Demo for Fact Checker Submission
"""
import json
import time
from datetime import datetime
from app_config import config
from src.claim_extractor import ClaimExtractor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import FactRetriever
from src.llm_client import GroqClient
from src.fact_checker import FactChecker

class FactCheckerDemo:
    def __init__(self):
        self.system = None
        self.results = []
        
    def initialize_system(self):
        """Initialize the fact checking system"""
        print("🚀 Initializing Fact Checking System...")
        
        # Validate configuration
        config.validate_config()
        
        # Initialize components
        claim_extractor = ClaimExtractor()
        embedding_generator = EmbeddingGenerator(config.EMBEDDING_MODEL)
        vector_store = VectorStore(config.FAISS_INDEX_PATH, config.METADATA_PATH)
        
        # Load vector store
        vector_store.load()
        
        retriever = FactRetriever(embedding_generator, vector_store)
        llm_client = GroqClient(config.GROQ_API_KEY, config.GROQ_MODEL)
        self.system = FactChecker(claim_extractor, retriever, llm_client)
        
        print("✅ System initialized successfully!")
        
        # Show system info
        stats = vector_store.get_stats()
        print(f"📊 Vector Database: {stats['total_entries']} documents")
        print(f"🤖 Using Model: {config.GROQ_MODEL}")
        
    def run_comprehensive_demo(self):
        """Run a comprehensive demonstration"""
        print("\n" + "="*70)
        print("🎯 FACT CHECKER COMPREHENSIVE DEMONSTRATION")
        print("="*70)
        
        test_cases = [
            {
                "category": "✅ TRUE CLAIMS (Exact Matches)",
                "claims": [
                    "Indian Railways have earmarked Nilgiri Mountain Railways for promotion of heritage tourism by steam locomotive",
                    "Ministry of Rural Development has released Rs. 70.90 lakh to Assam for rural drinking water supply",
                    "SAIL intends to invest Rs. 800-1000 crore every year to upgrade existing facilities during 10th Plan period",
                    "There is a proposal for installation of IVth Coke Oven Battery at Visakhapatnam Steel Plant costing Rs. 303 crores"
                ]
            },
            {
                "category": "❌ FALSE CLAIMS", 
                "claims": [
                    "The government announced free electricity to all farmers starting July 2025",
                    "Indian Railways cancelled all heritage tourism programs",
                    "Ministry of Steel announced 50% salary hike for all employees",
                    "Government stopped all rural development funding"
                ]
            },
            {
                "category": "🔍 PARTIAL/MIXED CLAIMS",
                "claims": [
                    "Railways are focusing on steam locomotives for tourism",
                    "Rural development funds were allocated to northeastern states", 
                    "Steel plants are getting new investments",
                    "Government announced new policies for farmers"
                ]
            }
        ]
        
        all_results = []
        
        for test_case in test_cases:
            print(f"\n{test_case['category']}")
            print("-" * 50)
            
            for claim in test_case['claims']:
                print(f"\n📝 Claim: {claim}")
                
                start_time = time.time()
                result = self.system.check_claim(claim)
                processing_time = time.time() - start_time
                
                # Add processing time to result
                result['processing_time'] = round(processing_time, 2)
                
                # Display result
                verdict_symbol = "✅ TRUE" if result['verdict'] == "TRUE" else "❌ FALSE" if result['verdict'] == "FALSE" else "❓ UNVERIFIABLE"
                print(f"🎯 {verdict_symbol} | ⏱️ {processing_time:.2f}s | 📊 Confidence: {result.get('confidence', 'N/A')}")
                print(f"💡 Reasoning: {result['reasoning'][:150]}...")
                
                if result.get('key_evidence'):
                    print(f"🔎 Evidence: {result['key_evidence'][0]}")
                
                all_results.append(result)
                
                # Brief pause between requests
                time.sleep(1)
        
        return all_results
    
    def generate_sample_output(self):
        """Generate sample input/output files for submission"""
        print("\n" + "="*70)
        print("📁 GENERATING SAMPLE FILES FOR SUBMISSION")
        print("="*70)
        
        sample_claims = [
            "The Indian government has announced free electricity to all farmers starting July 2025",
            "Indian Railways have earmarked Nilgiri Mountain Railways for promotion of heritage tourism by steam locomotive",
            "Ministry of Rural Development has released Rs. 70.90 lakh to Assam for rural drinking water supply",
            "SAIL intends to invest Rs. 800-1000 crore every year during 10th Plan period"
        ]
        
        sample_inputs = {"claims": sample_claims}
        sample_outputs = []
        
        print("Processing sample claims...")
        for claim in sample_claims:
            result = self.system.check_claim(claim)
            sample_outputs.append(result)
            print(f"✅ Processed: {claim[:50]}...")
        
        # Save sample files
        with open('sample_inputs.json', 'w', encoding='utf-8') as f:
            json.dump(sample_inputs, f, indent=2, ensure_ascii=False)
        
        with open('sample_outputs.json', 'w', encoding='utf-8') as f:
            json.dump(sample_outputs, f, indent=2, ensure_ascii=False)
        
        print("✅ Generated sample_inputs.json and sample_outputs.json")
        
        return sample_inputs, sample_outputs
    
    def system_analytics(self):
        """Show system performance analytics"""
        print("\n" + "="*70)
        print("📊 SYSTEM ANALYTICS")
        print("="*70)
        
        # Test a few claims to gather metrics
        test_claims = [
            "Indian Railways heritage tourism",
            "Rural development funds Assam", 
            "Steel plant investment",
            "Government new policy"
        ]
        
        processing_times = []
        
        for claim in test_claims:
            start_time = time.time()
            result = self.system.check_claim(claim)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            print(f"⏱️ '{claim[:30]}...': {processing_time:.2f}s")
        
        avg_time = sum(processing_times) / len(processing_times)
        print(f"\n📈 Average processing time: {avg_time:.2f}s")
        print(f"⚡ Fastest: {min(processing_times):.2f}s")
        print(f"🐢 Slowest: {max(processing_times):.2f}s")
        
        return {
            "average_processing_time": avg_time,
            "total_test_claims": len(test_claims),
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main demonstration function"""
    demo = FactCheckerDemo()
    
    try:
        # Initialize system
        demo.initialize_system()
        
        # Run comprehensive demo
        results = demo.run_comprehensive_demo()
        
        # Generate sample files
        demo.generate_sample_output()
        
        # Show analytics
        analytics = demo.system_analytics()
        
        print("\n" + "="*70)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("="*70)
        print("📁 Files generated for submission:")
        print("   • sample_inputs.json")
        print("   • sample_outputs.json") 
        print("   • fact_checker.log (if logging enabled)")
        print(f"\n📊 Performance: {analytics['average_processing_time']:.2f}s average")
        print("🚀 Ready for submission!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()