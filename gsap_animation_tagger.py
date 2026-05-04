#!/usr/bin/env python3
"""
PROMETHEUS GSAP ANIMATION LOGIC TAGGER
Deep semantic tagging for GSAP animation modules (behavior, not appearance)
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import glob

class GSAPAnimationTagger:
    """Tags GSAP animation modules for Prometheus retrieval system"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.modules = []
        
    def find_all_modules(self) -> List[str]:
        """Discover all animation module folders"""
        modules = []
        for item in os.listdir(self.repo_path):
            full_path = os.path.join(self.repo_path, item)
            if os.path.isdir(full_path) and item != '.git' and item != 'testing':
                # Check if it has an index.html or notes.md
                if os.path.exists(os.path.join(full_path, 'index.html')) or \
                   os.path.exists(os.path.join(full_path, 'notes.md')):
                    modules.append(full_path)
        return sorted(modules)
    
    def read_notes(self, module_path: str) -> str:
        """Read notes.md file for a module"""
        notes_path = os.path.join(module_path, 'notes.md')
        if os.path.exists(notes_path):
            try:
                with open(notes_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return ""
        return ""
    
    def read_html(self, module_path: str) -> str:
        """Read index.html file for a module"""
        html_path = os.path.join(module_path, 'index.html')
        if os.path.exists(html_path):
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return ""
        return ""
    
    def extract_animation_durations(self, html_content: str) -> Dict[str, Any]:
        """Extract timing information from GSAP code"""
        durations = {}
        
        # Find GSAP timeline durations
        duration_matches = re.findall(r'\.to\([^,]+,\s*\{[^}]*duration:\s*([0-9.]+)', html_content)
        delay_matches = re.findall(r'delay:\s*([0-9.]+)', html_content)
        
        total_estimated = 0
        if duration_matches:
            total_estimated = sum(float(d) for d in duration_matches) * 1000
        
        return {
            'estimatedDurationMs': int(total_estimated) if total_estimated else 800,
            'hasStagger': 'stagger' in html_content.lower(),
            'hasLoop': 'repeat' in html_content or 'yoyo' in html_content,
            'hasScrub': 'scrub' in html_content,
        }
    
    def extract_easing(self, html_content: str) -> List[str]:
        """Extract easing types from GSAP code"""
        eases = set()
        
        # Common GSAP eases
        ease_patterns = [
            'none', 'power1', 'power2', 'power3', 'power4',
            'back', 'elastic', 'bounce', 'circ', 'sine', 'expo', 'quad',
            'quad.inOut', 'quad.out', 'quad.in',
            'cubic', 'quart', 'quint', 'linear'
        ]
        
        html_lower = html_content.lower()
        for ease in ease_patterns:
            if f'ease:' in html_lower and ease in html_lower:
                eases.add(ease)
        
        return list(eases) if eases else ['quad', 'sine']
    
    def infer_primary_function(self, module_name: str, notes: str, html: str) -> str:
        """Infer the primary animation function"""
        name_lower = module_name.lower()
        notes_lower = notes.lower()
        
        # Check for specific patterns
        if 'reveal' in name_lower or 'reveal' in notes_lower:
            if 'hero' in name_lower or 'hero' in notes_lower:
                return 'hero_asset_reveal'
            elif 'text' in name_lower or 'mask' in name_lower:
                return 'text_mask_reveal'
            else:
                return 'cinematic_reveal'
        
        elif 'portrait' in name_lower or 'portrait' in notes_lower:
            return 'hero_asset_reveal'
        
        elif 'text' in name_lower and 'mask' in name_lower:
            return 'text_mask_reveal'
        
        elif 'transition' in name_lower or 'transition' in notes_lower:
            return 'cinematic_transition'
        
        elif 'beat' in name_lower or 'beat' in notes_lower:
            if 'rejection' in name_lower or 'danger' in name_lower:
                return 'emotional_payoff'
            elif 'authority' in name_lower or 'authority' in notes_lower:
                return 'authority_emphasis'
            elif 'card' in name_lower or 'content' in name_lower:
                return 'card_sequence'
            else:
                return 'cinematic_beat'
        
        elif 'carousel' in name_lower or 'carousel' in notes_lower or 'stack' in name_lower:
            return 'motion_graphic_holder'
        
        elif 'card' in name_lower or 'card' in notes_lower:
            return 'card_sequence'
        
        elif 'center' in name_lower and 'text' in name_lower:
            return 'text_emphasis'
        
        elif 'typewriter' in name_lower or 'type' in name_lower:
            return 'narrative_text_reveal'
        
        else:
            return 'static_asset_choreography'
    
    def infer_supported_assets(self, module_name: str, notes: str, html: str) -> List[str]:
        """Determine what asset types this animation supports"""
        assets = []
        
        content = (module_name + ' ' + notes + ' ' + html).lower()
        
        if 'text' in content or 'headline' in content or 'title' in content:
            assets.append('text')
            assets.append('headline')
        
        if 'image' in content or 'portrait' in content or 'svg' in content:
            assets.append('static_image')
            assets.append('svg')
        
        if 'card' in content or 'component' in content:
            assets.append('card_component')
            assets.append('html_component')
        
        if 'logo' in content or 'icon' in content:
            assets.append('logo')
            assets.append('icon')
        
        if 'motion' in content or 'graphic' in content:
            assets.append('motion_graphic')
        
        if 'video' in content:
            assets.append('video_layer')
        
        # Add defaults if empty
        if not assets:
            assets = ['static_image', 'svg', 'text']
        
        return list(set(assets))
    
    def extract_motion_grammar(self, module_name: str, notes: str, html: str) -> List[str]:
        """Extract detailed motion grammar patterns"""
        grammar = set()
        content = (module_name + ' ' + notes + ' ' + html).lower()
        
        # Check for specific motion patterns
        if 'blur' in content:
            grammar.add('blur_to_clarity')
        if 'scale' in content or 'zoom' in content:
            grammar.add('scale_up_reveal')
        if 'slide' in content or 'enter' in content:
            if 'bottom' in content or 'up' in content:
                grammar.add('slide_in_from_bottom')
            elif 'left' in content:
                grammar.add('slide_in_from_left')
            elif 'right' in content:
                grammar.add('slide_in_from_right')
        if 'drop' in content or 'fall' in content:
            grammar.add('drop_from_top')
        if 'rotate' in content or 'orbit' in content or 'circular' in content:
            grammar.add('orbital_motion')
        if 'float' in content or 'hover' in content or 'hover' in content:
            grammar.add('floating_hover')
        if 'parallax' in content or 'depth' in content:
            grammar.add('parallax_drift')
        if 'mask' in content or 'clip' in content:
            grammar.add('mask_reveal')
        if 'underline' in content or 'draw' in content:
            grammar.add('underline_draw')
        if 'glow' in content or 'shine' in content:
            grammar.add('glow_pulse')
        if 'stagger' in content or 'sequence' in content:
            grammar.add('kinetic_stagger')
        if 'elastic' in content or 'bounce' in content:
            grammar.add('elastic_snap')
        if 'ease' in content or 'smooth' in content:
            grammar.add('premium_smooth_ease')
        if 'hierarchy' in content or 'layering' in content:
            grammar.add('subject_background_layering')
        if 'focus' in content or 'center' in content:
            grammar.add('hero_center_lock')
        if 'typewriter' in content or 'type' in content:
            grammar.add('character_sequential_reveal')
        if 'lower third' in content or 'lower-third' in content:
            grammar.add('lower_third_reveal')
        
        # Default motion grammar
        if not grammar:
            grammar.add('premium_smooth_ease')
            grammar.add('scale_up_reveal')
        
        return sorted(list(grammar))
    
    def analyze_notes_for_meaning(self, notes: str) -> Dict[str, List[str]]:
        """Extract semantic meaning from notes.md"""
        meaning = {
            'rhetorical': [],
            'emotional': [],
            'use_cases': [],
            'symbolic': [],
            'best_for': [],
            'avoid_when': [],
        }
        
        notes_lower = notes.lower()
        
        # Extract best use cases from "Best Used For" or "Best used for" section
        best_match = re.search(r'best used? for:?(.*?)(?:---|\Z)', notes, re.IGNORECASE | re.DOTALL)
        if best_match:
            best_text = best_match.group(1)
            # Extract bullet points
            items = re.findall(r'[-*]\s*([^\n]+)', best_text)
            meaning['best_for'].extend([item.strip() for item in items][:6])
        
        # Extract avoid patterns
        avoid_match = re.search(r'avoid.*?(?:using|when).*?:(.*?)(?:---|\Z)', notes, re.IGNORECASE | re.DOTALL)
        if avoid_match:
            avoid_text = avoid_match.group(1)
            items = re.findall(r'[-*]\s*([^\n]+)', avoid_text)
            meaning['avoid_when'].extend([item.strip() for item in items][:5])
        
        # Semantic inference
        if 'authority' in notes_lower:
            meaning['rhetorical'].append('authority')
            meaning['symbolic'].extend(['authority', 'dominance', 'control'])
        
        if 'premium' in notes_lower or 'luxury' in notes_lower or 'elegant' in notes_lower:
            meaning['emotional'].append('premium')
            meaning['symbolic'].extend(['luxury', 'sophistication', 'excellence'])
        
        if 'founder' in notes_lower or 'creator' in notes_lower:
            meaning['use_cases'].append('creator_branding')
            meaning['use_cases'].append('founder_intro')
        
        if 'emotional' in notes_lower or 'emotion' in notes_lower:
            meaning['emotional'].append('emotional_resonance')
        
        if 'hero' in notes_lower:
            meaning['rhetorical'].append('hook')
            meaning['symbolic'].append('prominence')
        
        if 'proof' in notes_lower or 'evidence' in notes_lower:
            meaning['rhetorical'].append('proof')
            meaning['symbolic'].append('credibility')
        
        if 'transition' in notes_lower or 'scene' in notes_lower:
            meaning['rhetorical'].append('transition')
        
        if 'subtle' in notes_lower or 'restrained' in notes_lower:
            meaning['emotional'].append('calm')
        
        if 'aggressive' in notes_lower or 'intense' in notes_lower or 'punchy' in notes_lower:
            meaning['emotional'].append('intensity')
        
        # Add defaults
        if not meaning['rhetorical']:
            meaning['rhetorical'] = ['emphasis', 'focus']
        if not meaning['emotional']:
            meaning['emotional'] = ['clarity', 'professionalism']
        if not meaning['symbolic']:
            meaning['symbolic'] = ['progress', 'motion', 'clarity', 'structure']
        
        return meaning
    
    def generate_vector_search_text(self, module_name: str, metadata: Dict) -> str:
        """Create comprehensive vector search text"""
        components = [
            module_name,
            metadata.get('primaryAnimationFunction', ''),
            ' '.join(metadata.get('rhetoricalRoles', [])[:3]),
            ' '.join(metadata.get('emotionalRoles', [])[:3]),
            ' '.join(metadata.get('motionGrammar', [])[:4]),
            ' '.join(metadata.get('supportedAssetTypes', [])[:3]),
        ]
        
        text = ' '.join(filter(None, components))
        
        # Add semantic richness
        if 'reveal' in metadata.get('primaryAnimationFunction', ''):
            text += ' hero entrance cinematic disclosure'
        if 'hero' in text:
            text += ' premium authority emotional hook'
        if 'text' in text:
            text += ' narrative typography emphasis impact'
        if 'blur' in text:
            text += ' focus isolation clarity resolution'
        
        return text[:300]
    
    def tag_module(self, module_path: str) -> Optional[Dict[str, Any]]:
        """Generate complete metadata for a GSAP module"""
        try:
            module_name = os.path.basename(module_path)
            notes = self.read_notes(module_path)
            html = self.read_html(module_path)
            
            # Extract base information
            timing = self.extract_animation_durations(html)
            easing = self.extract_easing(html)
            primary_fn = self.infer_primary_function(module_name, notes, html)
            supported_assets = self.infer_supported_assets(module_name, notes, html)
            motion_grammar = self.extract_motion_grammar(module_name, notes, html)
            meaning = self.analyze_notes_for_meaning(notes)
            
            # Confidence score based on available data
            confidence = 0.7  # Base confidence
            if notes:
                confidence += 0.2
            if len(html) > 1000:
                confidence += 0.1
            confidence = min(confidence, 1.0)
            
            metadata = {
                "moduleId": module_name.replace(' ', '_').lower(),
                "moduleName": module_name,
                "relativePath": f"/{module_name}",
                "assetType": "gsap_animation_logic",
                
                "primaryAnimationFunction": primary_fn,
                "secondaryAnimationFunctions": [
                    'hierarchy_reinforcement',
                    'focus_isolation',
                    'premium_perception',
                ],
                
                "rhetoricalRoles": meaning['rhetorical'] + ['emphasis', 'engagement'],
                "emotionalRoles": meaning['emotional'] + ['clarity', 'momentum'],
                "sceneUseCases": meaning['use_cases'][:5],
                
                "motionGrammar": motion_grammar,
                
                "timingProfile": {
                    "estimatedDurationMs": timing['estimatedDurationMs'],
                    "tempo": "moderate" if timing['estimatedDurationMs'] < 1200 else "deliberate",
                    "pacingShape": "ease_out" if timing['estimatedDurationMs'] < 800 else "complex",
                    "staggered": timing['hasStagger'],
                    "looping": timing['hasLoop'],
                    "scrubbed": timing['hasScrub'],
                },
                
                "easingProfile": {
                    "detectedEases": easing,
                    "easePersonality": "premium_smooth" if any('power' in e or 'sine' in e for e in easing) else "elastic",
                    "premiumFeel": "high" if 'power' in str(easing) else "medium",
                },
                
                "animationPhases": [
                    {
                        "phase": "entrance",
                        "description": "Asset enters from reduced opacity and scale with blur",
                        "motionGrammar": motion_grammar[:2] if motion_grammar else ["blur_to_clarity"],
                        "durationEstimateMs": int(timing['estimatedDurationMs'] * 0.4),
                    },
                    {
                        "phase": "emphasis_hold",
                        "description": "Asset settles into focus position with subtle motion",
                        "motionGrammar": motion_grammar[1:3] if len(motion_grammar) > 1 else ["floating_hover"],
                        "durationEstimateMs": int(timing['estimatedDurationMs'] * 0.4),
                    },
                    {
                        "phase": "exit",
                        "description": "Asset exits or transitions to next element",
                        "motionGrammar": ["scale_down_settle", "fade_out"],
                        "durationEstimateMs": int(timing['estimatedDurationMs'] * 0.2),
                    }
                ],
                
                "supportedAssetTypes": supported_assets,
                "replaceableSlots": [
                    {
                        "slotName": "mainAsset",
                        "slotType": "image_or_svg",
                        "description": "Primary animated asset",
                        "recommendedReplacements": supported_assets,
                        "constraints": ["centered_composition_preferred"],
                    }
                ] if any(t in supported_assets for t in ['static_image', 'svg', 'logo']) else [],
                
                "dynamicInputs": [
                    "imageUrl", "assetScale", "duration", "easing", "accentColor"
                ] if 'text' not in ' '.join(supported_assets) else ["headlineText", "subtitleText", "accentColor", "duration"],
                
                "layoutBehavior": {
                    "recommendedPlacement": ["center_hero", "full_frame_transition"],
                    "safeZones": ["center_frame", "lower_third"],
                    "aspectRatioFit": ["16:9", "9:16", "1:1"],
                    "depthLayering": "strong_depth_illusion" if 'parallax' in str(motion_grammar) else "flat_overlay",
                    "foregroundMidgroundBackground": ["hero_asset", "supporting_text", "background_texture"],
                },
                
                "compatibility": {
                    "worksWithStaticImages": 'static_image' in supported_assets,
                    "worksWithMotionGraphics": 'motion_graphic' in supported_assets,
                    "worksWithTypography": 'text' in supported_assets or 'headline' in supported_assets,
                    "worksWithSVG": 'svg' in supported_assets,
                    "worksWithVideo": 'video_layer' in supported_assets,
                    "requiresTransparentAsset": 'blur' in str(motion_grammar) or 'mask' in str(motion_grammar),
                    "requiresMatting": 'mask' in str(motion_grammar),
                    "supportsBehindSubjectText": 'subject_background_layering' in motion_grammar,
                },
                
                "negativeGrammar": {
                    "forbiddenPairings": meaning['avoid_when'],
                    "riskFactors": [
                        "can_feel_repetitive_if_overused",
                        "performance_intensive_on_mobile",
                        "requires_transparent_asset" if 'mask' in str(motion_grammar) else "none",
                    ],
                    "avoidWhen": [
                        "scene_is_visually_busy",
                        "previous_beat_used_same_motion",
                        "moment_requires_restraint",
                    ] + meaning['avoid_when'],
                    "overuseRisk": "medium" if 'reveal' in primary_fn else "low",
                },
                
                "judgmentEngineHints": {
                    "bestWhen": meaning['best_for'],
                    "avoidWhen": meaning['avoid_when'],
                    "retrievalTriggers": [
                        f"primaryFunction={primary_fn}",
                        "emotionalIntensity=premium",
                        "requiresPremium=true",
                    ],
                    "sequenceRole": "opener" if 'hero' in primary_fn else "transition",
                    "intensityLevel": "moderate",
                    "noveltyLevel": "medium",
                    "restraintLevel": "medium",
                },
                
                "renderProfile": {
                    "renderComplexity": "high" if any(x in str(motion_grammar) for x in ['blur', 'parallax', 'mask']) else "medium",
                    "browserPreviewRisk": "low",
                    "remotionRenderRisk": "medium" if 'blur' in str(motion_grammar) else "low",
                    "performanceNotes": [
                        "Uses GSAP timeline",
                        "Blur filter applied" if 'blur' in str(motion_grammar) else "No heavy filters",
                    ],
                },
                
                "styleFamily": ["cinematic_premium", "modern_web_motion", "apple_style"],
                "creatorFit": ["high_ticket_founder", "premium_podcast_editor", "SaaS_founder_content"],
                "symbolicMeaning": meaning['symbolic'] + ["excellence", "precision", "execution"],
                
                "tags": {
                    "literalTags": [module_name, primary_fn],
                    "semanticTags": meaning['rhetorical'] + meaning['emotional'] + meaning['symbolic'],
                    "motionTags": motion_grammar,
                    "rhetoricalTags": meaning['rhetorical'],
                    "technicalTags": ["gsap", "timeline", "javascript"] + easing,
                    "retrievalTags": [primary_fn] + motion_grammar[:3] + supported_assets[:2],
                },
                
                "vectorSearchText": self.generate_vector_search_text(module_name, {
                    'primaryAnimationFunction': primary_fn,
                    'rhetoricalRoles': meaning['rhetorical'],
                    'emotionalRoles': meaning['emotional'],
                    'motionGrammar': motion_grammar,
                    'supportedAssetTypes': supported_assets,
                }),
                
                "confidenceScore": confidence,
                "confidenceNotes": f"Based on {'notes.md + ' if notes else ''}HTML code analysis",
            }
            
            return metadata
            
        except Exception as e:
            print(f"✗ Error tagging {module_path}: {e}")
            return None
    
    def process_all_modules(self) -> List[Dict[str, Any]]:
        """Process all GSAP modules"""
        modules = self.find_all_modules()
        print(f"Found {len(modules)} GSAP animation modules\n")
        
        metadata_list = []
        for module_path in modules:
            module_name = os.path.basename(module_path)
            metadata = self.tag_module(module_path)
            if metadata:
                metadata_list.append(metadata)
                print(f"✓ Tagged: {module_name:50} | {metadata['primaryAnimationFunction']}")
        
        return metadata_list
    
    def save_metadata(self, metadata_list: List[Dict], output_path: str):
        """Save metadata to JSON"""
        output = {
            "system": "PROMETHEUS_GSAP_ANIMATION_TAGGING",
            "version": "1.0",
            "repo": "GSAP_ANIMATION",
            "taggingMethod": "Notes.md + Code Inspection + Animation Grammar Analysis",
            "assetCount": len(metadata_list),
            "modules": metadata_list
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Metadata saved to: {output_path}")
        print(f"✓ Total modules tagged: {len(metadata_list)}")
    
    def create_summary(self, metadata_list: List[Dict], output_path: str):
        """Create summary markdown"""
        primary_functions = {}
        asset_types = {}
        motion_grammars = set()
        
        for meta in metadata_list:
            pf = meta['primaryAnimationFunction']
            primary_functions[pf] = primary_functions.get(pf, 0) + 1
            
            for asset in meta['supportedAssetTypes']:
                asset_types[asset] = asset_types.get(asset, 0) + 1
            
            for grammar in meta['motionGrammar']:
                motion_grammars.add(grammar)
        
        summary = f"""# PROMETHEUS GSAP ANIMATION TAGGING SUMMARY

## Overview

**Total Modules Tagged:** {len(metadata_list)}
**Tagging Method:** Notes.md + Code Inspection + Animation Grammar Analysis
**Repository:** GSAP_ANIMATION (github.com/igrisundead-ops/GSAP_ANIMATION)

---

## Primary Animation Functions

Distribution of primary animation functions across modules:

"""
        
        for pf, count in sorted(primary_functions.items(), key=lambda x: x[1], reverse=True):
            summary += f"- **{pf}**: {count} modules\n"
        
        summary += f"""

---

## Supported Asset Types

Animation modules support the following asset types:

"""
        
        for asset, count in sorted(asset_types.items(), key=lambda x: x[1], reverse=True):
            summary += f"- **{asset}**: {count} modules\n"
        
        summary += f"""

---

## Motion Grammar Patterns

**Total Unique Motion Patterns:** {len(motion_grammars)}

"""
        
        for grammar in sorted(motion_grammars):
            summary += f"- {grammar}\n"
        
        summary += """

---

## Key Insights

### Premium Animation Behaviors

The GSAP modules emphasize:

1. **Cinematic Reveals** - Premium blur-to-clarity, scale-up entrances for hero assets
2. **Text Emphasis** - Masked reveals, character-sequential animation, typographic hierarchy
3. **Subject Focus** - Portrait reveals, layering strategies for depth illusion
4. **Restraint** - Smooth easing, moderate pacing, avoiding aggressive motion
5. **Reusability** - Modular GSAP timelines, dynamic asset slots, configurable timing

### Creator Fit

All modules are designed for:

- High-ticket founders and creators
- Premium SaaS product demos
- Editorial/cinematic content
- Authority-driven branding
- Luxury lifestyle positioning
- Sophisticated creator content

### Render Complexity

- **Low Complexity:** Text animations, simple reveal transitions
- **Medium Complexity:** Multi-asset choreography, parallax effects
- **High Complexity:** Masked reveals, heavy blur effects, complex stagger patterns

---

## Retrieval Strategy for Prometheus Judgment Engine

The tagged metadata enables retrieval via:

1. **Vector Search** - Rich vectorSearchText fields for semantic matching
2. **Motion Grammar Tags** - Specific motion behavior patterns
3. **Asset Compatibility** - Which assets each animation supports
4. **Rhetorical Roles** - Hook, proof, authority, transition, payoff
5. **Emotional Intensity** - Restraint level, intensity level, novelty level
6. **Creator Fit** - Target creator type and content style

---

Generated: PROMETHEUS GSAP Animation Intelligence System
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"✓ Summary created: {output_path}")
    
    def create_validation_report(self, metadata_list: List[Dict], output_path: str):
        """Create validation report"""
        
        missing_notes = []
        weak_metadata = []
        unknown_duration = []
        unclear_slots = []
        text_only = []
        supports_static = []
        supports_motion = []
        high_render_risk = []
        duplicates = {}
        
        for meta in metadata_list:
            name = meta['moduleName']
            
            if meta['confidenceScore'] < 0.6:
                weak_metadata.append(name)
            
            if meta['timingProfile']['estimatedDurationMs'] == 800:
                unknown_duration.append(name)
            
            if not meta['replaceableSlots']:
                unclear_slots.append(name)
            
            if meta['supportedAssetTypes'] == ['text', 'headline', 'subtitle']:
                text_only.append(name)
            
            if meta['compatibility']['worksWithStaticImages']:
                supports_static.append(name)
            
            if meta['compatibility']['worksWithMotionGraphics']:
                supports_motion.append(name)
            
            if meta['renderProfile']['renderComplexity'] == 'high':
                high_render_risk.append(name)
            
            # Check for duplicates
            pf = meta['primaryAnimationFunction']
            if pf not in duplicates:
                duplicates[pf] = []
            duplicates[pf].append(name)
        
        report = f"""# PROMETHEUS GSAP ANIMATION TAGGING VALIDATION REPORT

## Summary

- **Total Modules Processed:** {len(metadata_list)}
- **Modules With Complete Metadata:** {len([m for m in metadata_list if m['confidenceScore'] >= 0.8])}
- **Modules Needing Review:** {len(weak_metadata)}

---

## Quality Issues

### Weak Metadata ({len(weak_metadata)} modules)

Modules with confidence score < 0.6 (may need review or expanded notes):

"""
        for item in weak_metadata:
            report += f"- {item}\n"
        
        report += f"""

### Unknown Duration ({len(unknown_duration)} modules)

Modules where duration could not be extracted from code:

"""
        for item in unknown_duration[:10]:
            report += f"- {item}\n"
        
        report += f"""

### Unclear Replaceable Slots ({len(unclear_slots)} modules)

Modules without identified swappable asset slots:

"""
        for item in unclear_slots[:10]:
            report += f"- {item}\n"
        
        report += f"""

### Text-Only Animations ({len(text_only)} modules)

Modules that only support typography:

"""
        for item in text_only:
            report += f"- {item}\n"
        
        report += f"""

---

## Asset Compatibility

### Support Static Images ({len(supports_static)} modules)

```
{len(supports_static)} of {len(metadata_list)} modules can animate static images
```

### Support Motion Graphics ({len(supports_motion)} modules)

```
{len(supports_motion)} of {len(metadata_list)} modules can hold motion graphics as nested content
```

---

## Performance Assessment

### High Render Complexity ({len(high_render_risk)} modules)

Modules that may impact render performance:

"""
        for item in high_render_risk[:10]:
            report += f"- {item}\n"
        
        report += f"""

---

## Pattern Analysis

### Animation Function Duplicates

Functions with multiple implementations:

"""
        for func, modules in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True):
            if len(modules) > 1:
                report += f"- **{func}** ({len(modules)} variations): {', '.join(modules[:3])}\n"
        
        report += """

---

## Recommendations

### Priority Review Items

1. **Add More Detailed Notes** - Modules with confidence < 0.8 should have expanded notes.md files
2. **Clarify Asset Slots** - Modules without replaceable slots may need better documentation
3. **Extract Duration Data** - Timing information helps with scene pacing decisions

### Optimization Opportunities

- Consider consolidating duplicate animation patterns
- Create animation "templates" from high-confidence modules
- Document performance characteristics for blur/mask/parallax modules

### Best Practices

- Always reference the vectorSearchText field for semantic retrieval
- Use motionGrammar tags for motion-based filtering
- Check compatibility flags before assigning assets
- Review negativeGrammar to avoid poor pairings

---

Generated: PROMETHEUS Validation System
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Validation report created: {output_path}")


def main():
    repo_path = r"c:\Users\HomePC\Downloads\MATT MANHUNTER\GSAP_REPO"
    output_json = r"c:\Users\HomePC\Downloads\MATT MANHUNTER\gsap-animation-metadata.json"
    output_summary = r"c:\Users\HomePC\Downloads\MATT MANHUNTER\GSAP_TAGGING_SUMMARY.md"
    output_validation = r"c:\Users\HomePC\Downloads\MATT MANHUNTER\GSAP_TAGGING_VALIDATION.md"
    
    tagger = GSAPAnimationTagger(repo_path)
    metadata_list = tagger.process_all_modules()
    
    if metadata_list:
        tagger.save_metadata(metadata_list, output_json)
        tagger.create_summary(metadata_list, output_summary)
        tagger.create_validation_report(metadata_list, output_validation)
        print("\n✓ All tagging complete!")
    else:
        print("✗ No modules were successfully tagged")


if __name__ == "__main__":
    main()
