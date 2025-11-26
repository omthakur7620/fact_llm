"""
Groq API client with enhanced semantic understanding for fact-checking
"""
import os
import requests
import json
from typing import Dict, Any, List

class GroqClient:
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def fact_check_claim(self, claim: str, retrieved_facts: List[Dict]) -> Dict[str, Any]:
        """
        Use LLM to fact-check with enhanced semantic understanding
        """
        print(f"🔍 Fact-checking: '{claim}'")
        print(f"📚 Retrieved {len(retrieved_facts)} documents")
        
        prompt = self._build_semantic_understanding_prompt(claim, retrieved_facts)
        
        try:
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 800,
                "top_p": 0.9
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()
            result_text = response_data["choices"][0]["message"]["content"].strip()
            
            print(f"📄 LLM Response: {result_text[:200]}...")
            return self._parse_enhanced_response(result_text, retrieved_facts)
            
        except Exception as e:
            print(f"❌ API error: {e}")
            return self._get_error_response(str(e))
    
    def _build_semantic_understanding_prompt(self, claim: str, retrieved_facts: List[Dict]) -> str:
        """Build prompt that focuses on semantic understanding and conceptual matching"""
        
        # Format evidence for better readability
        evidence_text = ""
        if retrieved_facts:
            evidence_text = "## RELEVANT GOVERNMENT DOCUMENTS:\n\n"
            for i, fact in enumerate(retrieved_facts):
                evidence_text += f"""**Document {i+1}** (Similarity: {fact['similarity']:.3f})
**Source**: {fact['source']}
**Content**: {fact['content']}

"""
        else:
            evidence_text = "## RELEVANT GOVERNMENT DOCUMENTS:\nNo closely matching documents found.\n"

        return f"""# GOVERNMENT FACT-CHECKING TASK

## CLAIM TO VERIFY:
"{claim}"

{evidence_text}
## ANALYSIS INSTRUCTIONS:

You are a government fact-checker. Analyze if the claim is supported by the evidence.

### KEY CONCEPTUAL MATCHES TO LOOK FOR:

1. **ENTITY MATCHES**:
   - Same ministries, departments, organizations
   - Same locations, states, regions  
   - Same programs, schemes, initiatives
   - Same companies, public sector units

2. **THEMATIC MATCHES**:
   - Similar policy areas (tourism, rural development, steel, etc.)
   - Related government priorities
   - Comparable timeframes or periods
   - Similar types of announcements

3. **SEMANTIC EQUIVALENCE**:
   - "Steam locomotives" = "heritage tourism by steam locomotive"
   - "Released funds" = "released amount", "allocated money"
   - "Investing in upgrades" = "intends to invest", "upgradation"
   - "New equipment" = "installation", "new battery", "facilities"

### VERIFICATION GUIDELINES:

**✅ TRUE**: Evidence directly supports the claim through:
- Exact entity matches (same ministry, organization, location)
- Same core action or policy
- Matching financial amounts or timelines
- Logical inference from related government actions

**✅ LIKELY TRUE**: Strong conceptual alignment:
- Same thematic area with supporting evidence
- Related government initiatives that imply the claim
- Semantic equivalence in different wording

**❌ FALSE**: Clear contradiction with evidence

**❌ LIKELY FALSE**: Evidence suggests claim is incorrect

**🤷 UNVERIFIABLE**: No relevant evidence found

### EXAMPLES OF VALID INFERENCES:
- Claim: "Railways have steam locomotives for tourism" 
  Evidence: "Railways earmarked heritage tourism by steam locomotive" → ✅ TRUE

- Claim: "Rural development funds for Assam"
  Evidence: "Released funds to Assam for rural drinking water" → ✅ TRUE

- Claim: "Steel plant getting investments"
  Evidence: "SAIL intends to invest in upgradation" → ✅ TRUE

## YOUR ANALYSIS:

Focus on semantic meaning, not exact wording. Government documents use formal language while claims may use everyday terms.

Return ONLY JSON:

{{
    "verdict": "TRUE|LIKELY TRUE|FALSE|LIKELY FALSE|UNVERIFIABLE",
    "confidence": "high|medium|low",
    "reasoning": "Explain the semantic relationships found. Which entities match? Which concepts align?",
    "key_evidence": ["Specific matching evidence from documents"],
    "semantic_matches": {{
        "entity_matches": ["List matching entities"],
        "conceptual_alignment": "Describe thematic alignment",
        "wording_differences": "Note any terminology differences"
    }}
}}"""
    
    def _parse_enhanced_response(self, response_text: str, retrieved_facts: List[Dict]) -> Dict[str, Any]:
        """Parse the enhanced LLM response"""
        try:
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx+1]
                result = json.loads(json_str)
                
                # Ensure all required fields
                required = ['verdict', 'confidence', 'reasoning', 'key_evidence']
                for field in required:
                    if field not in result:
                        result[field] = "unknown" if field != 'key_evidence' else []
                
                # Add semantic_matches if missing
                if 'semantic_matches' not in result:
                    result['semantic_matches'] = {
                        'entity_matches': [],
                        'conceptual_alignment': 'not_analyzed',
                        'wording_differences': 'not_analyzed'
                    }
                
                # Add retrieval info
                result["retrieved_facts_count"] = len(retrieved_facts)
                result["retrieved_facts"] = retrieved_facts
                
                print(f"✅ Verdict: {result['verdict']} (Confidence: {result['confidence']})")
                return result
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"❌ Parse error: {e}")
            return self._get_fallback_response(retrieved_facts, str(e))
    
    def _get_fallback_response(self, retrieved_facts: List[Dict], error_msg: str) -> Dict[str, Any]:
        """Fallback when parsing fails"""
        return {
            "verdict": "UNVERIFIABLE",
            "confidence": "low",
            "reasoning": f"Analysis error: {error_msg}",
            "key_evidence": [],
            "semantic_matches": {
                "entity_matches": [],
                "conceptual_alignment": "analysis_failed",
                "wording_differences": "analysis_failed"
            },
            "retrieved_facts_count": len(retrieved_facts),
            "retrieved_facts": retrieved_facts
        }
    
    def _get_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Error response for API failures"""
        return {
            "verdict": "UNVERIFIABLE",
            "confidence": "very_low",
            "reasoning": f"Service error: {error_msg}",
            "key_evidence": [],
            "semantic_matches": {
                "entity_matches": [],
                "conceptual_alignment": "service_unavailable",
                "wording_differences": "service_unavailable"
            },
            "retrieved_facts_count": 0,
            "retrieved_facts": []
        }