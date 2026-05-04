#!/usr/bin/env python3
"""
PROMETHEUS MOTION GRAPHICS SEMANTIC TAGGER
Advanced deep semantic tagging for motion graphics assets
Uses cinematic analysis, not literal labeling
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any

class MotionGraphicsTagger:
    """Applies Prometheus semantic tagging to motion graphics assets"""
    
    def __init__(self, assets_dir: str):
        self.assets_dir = assets_dir
        self.metadata_list = []
        
    def analyze_filename(self, filename: str) -> Dict[str, Any]:
        """Extract semantic patterns from filename"""
        name = filename.replace('.html', '').lower()
        
        # Pattern analysis - optimized with hierarchical matching
        patterns = {
            'social': any(x in name for x in ['facebook', 'instagram', 'twitter', 'linkedin', 'youtube']),
            'step': 'step' in name or 'steps' in name or 'counter' in name or 'pointer' in name,
            'reveal': 'reveal' in name or 'blur' in name,
            'quote': 'quote' in name,
            'text': 'text' in name or 'word' in name or 'typewriter' in name or 'underline' in name or 'mask' in name or 'gradient' in name or 'highlight' in name or 'glow box' in name,
            'number': 'counter' in name or 'number' in name or 'count' in name,
            'card': 'card' in name,
            'transition': 'transition' in name or 'move' in name or 'journey' in name,
            'list': 'list' in name or 'check' in name or 'mark' in name,
            'graph': 'graph' in name or 'bar' in name or 'percentage' in name,
            'growth': 'growth' in name,
            'emphasis': 'highlight' in name or 'glow' in name or 'important' in name,
            'process': 'process' in name or 'systems' in name or 'loading' in name or 'calling' in name or 'message' in name,
            'people': 'person' in name or 'people' in name,
            'concept': 'idea' in name or 'information' in name or 'concept' in name or 'core' in name or 'main' in name,
            'interaction': 'interaction' in name or 'cursor' in name or 'search' in name or 'input' in name or 'selection' in name,
            'comparison': 'comparing' in name or 'two' in name or 'choice' in name or 'point' in name or 'comparing' in name,
            'authentication': 'selection' in name or 'choice' in name or 'check' in name,
            'hierarchy': 'pyramid' in name or 'petal' in name or 'bubble' in name,
            'curve': 'curve' in name,
            'date': 'date' in name,
            'money': 'money' in name or 'improvement' in name or 'chat' in name,
        }
        
        return patterns
    
    def analyze_file_content(self, filepath: str) -> Dict[str, Any]:
        """Analyze HTML file content for animation patterns"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().lower()
        except:
            return {}
        
        analysis = {
            'has_blur': 'blur' in content,
            'has_scale': 'scale' in content or 'transform' in content,
            'has_opacity': 'opacity' in content,
            'has_glow': 'glow' in content or 'shadow' in content or 'text-shadow' in content,
            'has_gradient': 'gradient' in content,
            'has_animation': 'animation' in content or '@keyframes' in content,
            'has_transition': 'transition' in content,
            'has_stagger': 'delay' in content or 'nth-child' in content,
            'has_canvas': 'canvas' in content,
            'has_svg': 'svg' in content,
            'is_responsive': 'vmin' in content or '100vw' in content or 'responsive' in content,
        }
        
        # Timing analysis
        animation_match = re.search(r'(\d+\.?\d*)\s*s(?:ec)?', content)
        duration_ms = int(float(animation_match.group(1)) * 1000) if animation_match else 800
        analysis['duration_ms'] = duration_ms
        
        return analysis
    
    def generate_semantic_tags(self, filename: str, patterns: Dict, analysis: Dict) -> Dict[str, List[str]]:
        """Generate deep semantic tags based on Prometheus system"""
        
        tags = {
            'primaryFunction': 'visual_element',
            'secondaryFunctions': [],
            'emotionalRoles': [],
            'rhetoricalRoles': [],
            'symbolism': [],
            'motionBehaviors': [],
            'styleFamily': [],
            'creatorFit': [],
            'sceneUseCases': [],
        }
        
        # HIERARCHICAL PRIMARY FUNCTION LOGIC - check in priority order
        if patterns.get('social'):
            tags['primaryFunction'] = 'social_proof_icon'
            tags['rhetoricalRoles'] = ['credibility', 'reach', 'validation']
            tags['emotionalRoles'] = ['trust', 'belonging', 'authority']
            tags['symbolism'] = ['influence', 'audience', 'reach']
            tags['sceneUseCases'] = ['credibility_signal', 'platform_mention', 'social_validation']
        
        elif patterns.get('quote'):
            tags['primaryFunction'] = 'testimony_showcase'
            tags['rhetoricalRoles'] = ['proof', 'social_proof', 'authority']
            tags['emotionalRoles'] = ['resonance', 'trust', 'inspiration']
            tags['symbolism'] = ['wisdom', 'authority', 'validation']
            tags['sceneUseCases'] = ['social_proof', 'quote_reveal', 'belief_confirmation']
        
        elif patterns.get('step'):
            tags['primaryFunction'] = 'procedural_disclosure'
            tags['rhetoricalRoles'] = ['progression', 'authority', 'instruction']
            tags['emotionalRoles'] = ['trust', 'clarity', 'momentum']
            tags['symbolism'] = ['progression', 'hierarchy', 'sequence', 'mastery']
            tags['sceneUseCases'] = ['sales_funnel', 'tutorial', 'process_explanation', 'credential_building']
        
        elif patterns.get('reveal'):
            tags['primaryFunction'] = 'cinematic_disclosure'
            tags['rhetoricalRoles'] = ['emphasis', 'discovery', 'tension_release']
            tags['emotionalRoles'] = ['anticipation', 'revelation', 'focus']
            tags['symbolism'] = ['breakthrough', 'clarity', 'emergence']
            tags['sceneUseCases'] = ['opening_beat', 'key_point_emphasis', 'value_disclosure']
        
        elif patterns.get('text'):
            tags['primaryFunction'] = 'narrative_delivery'
            tags['rhetoricalRoles'] = ['emphasis', 'pacing', 'retention']
            tags['emotionalRoles'] = ['engagement', 'believability', 'impact']
            tags['symbolism'] = ['authority', 'importance', 'message']
            tags['sceneUseCases'] = ['quote_revelation', 'headline', 'proof_statement', 'call_to_action']
        
        elif patterns.get('number'):
            tags['primaryFunction'] = 'proof_mechanism'
            tags['rhetoricalRoles'] = ['evidence', 'quantification', 'social_proof']
            tags['emotionalRoles'] = ['authority', 'specificity', 'trust']
            tags['symbolism'] = ['achievement', 'scale', 'momentum']
            tags['sceneUseCases'] = ['results_showcase', 'metric_reveal', 'achievement_highlight']
        
        elif patterns.get('graph'):
            tags['primaryFunction'] = 'proof_visualization'
            tags['rhetoricalRoles'] = ['evidence', 'pattern_recognition', 'trajectory']
            tags['emotionalRoles'] = ['confidence', 'clarity', 'momentum']
            tags['symbolism'] = ['growth', 'performance', 'improvement']
            tags['sceneUseCases'] = ['results_slide', 'performance_metric', 'improvement_proof']
        
        elif patterns.get('growth'):
            tags['primaryFunction'] = 'momentum_amplification'
            tags['rhetoricalRoles'] = ['transformation', 'breakthrough', 'acceleration']
            tags['emotionalRoles'] = ['inspiration', 'momentum', 'empowerment']
            tags['symbolism'] = ['ascension', 'improvement', 'breaking_limits']
            tags['sceneUseCases'] = ['transformation_story', 'results_showcase', 'motivation_beat']
        
        elif patterns.get('card'):
            tags['primaryFunction'] = 'information_architecture'
            tags['rhetoricalRoles'] = ['organization', 'hierarchy', 'comparison']
            tags['emotionalRoles'] = ['clarity', 'professionalism', 'trust']
            tags['symbolism'] = ['structure', 'importance', 'premium']
            tags['sceneUseCases'] = ['feature_highlight', 'benefit_listing', 'testimonial_frame']
        
        elif patterns.get('list'):
            tags['primaryFunction'] = 'credential_sequencing'
            tags['rhetoricalRoles'] = ['verification', 'accumulation', 'proof']
            tags['emotionalRoles'] = ['trust', 'completion', 'momentum']
            tags['symbolism'] = ['achievement', 'verification', 'progress']
            tags['sceneUseCases'] = ['checklist_reveal', 'qualification_proof', 'benefit_list']
        
        elif patterns.get('interaction'):
            tags['primaryFunction'] = 'engagement_trigger'
            tags['rhetoricalRoles'] = ['call_to_action', 'interaction', 'invitation']
            tags['emotionalRoles'] = ['engagement', 'participation', 'agency']
            tags['symbolism'] = ['action', 'response', 'participation']
            tags['sceneUseCases'] = ['cta_emphasis', 'input_highlight', 'interaction_prompt']
        
        elif patterns.get('concept'):
            tags['primaryFunction'] = 'idea_crystallization'
            tags['rhetoricalRoles'] = ['concept_connection', 'meaning_making', 'framework']
            tags['emotionalRoles'] = ['clarity', 'breakthrough', 'resonance']
            tags['symbolism'] = ['insight', 'understanding', 'revelation']
            tags['sceneUseCases'] = ['concept_intro', 'framework_reveal', 'idea_highlight']
        
        elif patterns.get('people'):
            tags['primaryFunction'] = 'social_connection'
            tags['rhetoricalRoles'] = ['relatable', 'human', 'authentic']
            tags['emotionalRoles'] = ['connection', 'empathy', 'identification']
            tags['symbolism'] = ['humanity', 'authenticity', 'relatability']
            tags['sceneUseCases'] = ['testimonial_intro', 'avatar_reveal', 'person_mention']
        
        elif patterns.get('comparison'):
            tags['primaryFunction'] = 'comparative_analysis'
            tags['rhetoricalRoles'] = ['comparison', 'differentiation', 'proof']
            tags['emotionalRoles'] = ['clarity', 'emphasis', 'understanding']
            tags['symbolism'] = ['contrast', 'choice', 'decision']
            tags['sceneUseCases'] = ['feature_comparison', 'before_after', 'option_presentation']
        
        elif patterns.get('process'):
            tags['primaryFunction'] = 'system_articulation'
            tags['rhetoricalRoles'] = ['explanation', 'structure', 'clarity']
            tags['emotionalRoles'] = ['confidence', 'understanding', 'trust']
            tags['symbolism'] = ['order', 'system', 'control']
            tags['sceneUseCases'] = ['process_explanation', 'system_intro', 'workflow_reveal']
        
        elif patterns.get('hierarchy'):
            tags['primaryFunction'] = 'hierarchical_visualization'
            tags['rhetoricalRoles'] = ['organization', 'structure', 'priority']
            tags['emotionalRoles'] = ['clarity', 'order', 'authority']
            tags['symbolism'] = ['hierarchy', 'priority', 'structure']
            tags['sceneUseCases'] = ['priority_display', 'structure_reveal', 'ranking_show']
        
        elif patterns.get('curve'):
            tags['primaryFunction'] = 'dynamic_motion'
            tags['rhetoricalRoles'] = ['elegance', 'flow', 'sophistication']
            tags['emotionalRoles'] = ['elegance', 'premium', 'sophistication']
            tags['symbolism'] = ['flow', 'grace', 'precision']
            tags['sceneUseCases'] = ['transition', 'elegant_reveal', 'premium_motion']
        
        elif patterns.get('money'):
            tags['primaryFunction'] = 'financial_proof'
            tags['rhetoricalRoles'] = ['evidence', 'transformation', 'success']
            tags['emotionalRoles'] = ['wealth', 'success', 'empowerment']
            tags['symbolism'] = ['prosperity', 'growth', 'success']
            tags['sceneUseCases'] = ['income_reveal', 'profit_display', 'transformation_proof']
        
        # SECONDARY FUNCTIONS & MOTION BEHAVIORS
        if analysis.get('has_blur'):
            tags['secondaryFunctions'].append('focus_isolation')
            tags['motionBehaviors'].append('blur_to_clarity')
            tags['emotionalRoles'].extend(['focus', 'direction'])
        
        if analysis.get('has_glow'):
            tags['secondaryFunctions'].append('emphasis_amplification')
            tags['motionBehaviors'].append('luminous_emergence')
            tags['emotionalRoles'].extend(['premium', 'importance', 'magic'])
        
        if analysis.get('has_scale'):
            tags['motionBehaviors'].append('scale_transformation')
            tags['emotionalRoles'].append('emphasis')
        
        if analysis.get('has_stagger'):
            tags['motionBehaviors'].append('sequential_revelation')
            tags['emotionalRoles'].append('pacing')
        
        if analysis.get('has_gradient'):
            tags['styleFamily'].append('premium_gradient')
            tags['emotionalRoles'].append('luxury')
        
        # TIMING-BASED INTENSITY
        duration = analysis.get('duration_ms', 800)
        if duration < 400:
            tags['emotionalRoles'].extend(['sharp', 'punchy', 'immediate'])
        elif duration > 1200:
            tags['emotionalRoles'].extend(['deliberate', 'cinematic', 'reflective'])
        else:
            tags['emotionalRoles'].extend(['balanced', 'professional'])
        
        # CREATOR FIT
        tags['creatorFit'] = ['premium_creator', 'high_ticket_founder', 'cinematic_editor', 'authority_builder']
        tags['styleFamily'] = ['premium_motion', 'cinematic_web', 'modern_professional']
        
        return tags
    
    def create_vector_search_text(self, filename: str, tags: Dict) -> str:
        """Create semantic search text for Milvus vector search"""
        primary = tags.get('primaryFunction', 'visual_element')
        if isinstance(primary, list):
            primary = primary[0] if primary else 'visual_element'
        
        components = [
            filename.replace('.html', ''),
            primary,
            ' '.join(tags.get('emotionalRoles', [])[:3]),
            ' '.join(tags.get('rhetoricalRoles', [])[:3]),
            ' '.join(tags.get('symbolism', [])[:3]),
        ]
        return ' '.join(filter(None, components))
    
    def tag_asset(self, filename: str, filepath: str) -> Dict[str, Any]:
        """Generate complete metadata for single asset"""
        
        patterns = self.analyze_filename(filename)
        analysis = self.analyze_file_content(filepath)
        tags = self.generate_semantic_tags(filename, patterns, analysis)
        
        asset_id = filename.replace('.html', '').replace(' ', '_').lower()
        
        metadata = {
            "assetId": asset_id,
            "assetName": filename.replace('.html', ''),
            "assetType": "motion_graphic",
            "primaryFunction": tags['primaryFunction'],
            "secondaryFunctions": list(set(tags['secondaryFunctions'])),
            "emotionalRoles": list(set(tags['emotionalRoles']))[:6],
            "rhetoricalRoles": list(set(tags['rhetoricalRoles']))[:5],
            "visualEnergy": "high" if analysis.get('duration_ms', 800) < 800 else "moderate",
            "motionBehavior": list(set(tags['motionBehaviors'])),
            "styleFamily": tags['styleFamily'],
            "creatorFit": tags['creatorFit'],
            "sceneUseCases": list(set(tags['sceneUseCases']))[:5],
            "symbolicMeaning": list(set(tags['symbolism'])),
            "renderComplexity": "high" if analysis.get('has_canvas') or analysis.get('has_svg') else "moderate",
            "recommendedPlacement": ["opening", "key_point", "transition", "closing"],
            "vectorSearchText": self.create_vector_search_text(filename, tags),
            "animationDuration": f"{analysis.get('duration_ms', 800)}ms",
            "features": {
                "blur_effect": analysis.get('has_blur', False),
                "glow_effect": analysis.get('has_glow', False),
                "staggered_animation": analysis.get('has_stagger', False),
                "responsive_design": analysis.get('is_responsive', False),
                "canvas_rendering": analysis.get('has_canvas', False),
            }
        }
        
        return metadata
    
    def process_all_assets(self) -> List[Dict[str, Any]]:
        """Process all HTML files in the assets directory"""
        metadata_list = []
        
        for filename in sorted(os.listdir(self.assets_dir)):
            if filename.endswith('.html'):
                filepath = os.path.join(self.assets_dir, filename)
                try:
                    metadata = self.tag_asset(filename, filepath)
                    metadata_list.append(metadata)
                    print(f"✓ Tagged: {filename}")
                except Exception as e:
                    print(f"✗ Error tagging {filename}: {e}")
        
        return metadata_list
    
    def save_metadata(self, metadata_list: List[Dict], output_path: str):
        """Save metadata to JSON file"""
        output = {
            "system": "PROMETHEUS_MOTION_GRAPHICS_TAGGING",
            "version": "2.0",
            "assetCount": len(metadata_list),
            "assets": metadata_list
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Metadata saved to: {output_path}")
        print(f"✓ Total assets tagged: {len(metadata_list)}")


def main():
    assets_dir = r"c:\Users\HomePC\Downloads\HELP, VIDEO MATTING\STRUCTURED ANIMATION"
    output_path = r"c:\Users\HomePC\Downloads\MATT MANHUNTER\MOTION_GRAPHICS_METADATA.json"
    
    tagger = MotionGraphicsTagger(assets_dir)
    metadata_list = tagger.process_all_assets()
    tagger.save_metadata(metadata_list, output_path)


if __name__ == "__main__":
    main()
