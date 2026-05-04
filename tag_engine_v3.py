#!/usr/bin/env python3
"""
Multi-layered Asset Tagging Engine v3 - CONTENT-AWARE + FOLDER REFINED
Analyzes image content and applies intelligent folder-context refinement.
"""

import os
import json
from pathlib import Path
from PIL import Image
import hashlib
from collections import defaultdict
import numpy as np

# FOLDER CONTEXTS
FOLDER_CONTEXTS = {
    "CLAUDE STATIC ABSTRACT SHAPES board - Apr 20th 2026 (11 images)": {
        "category": "abstract-design",
        "brand_context": "design-system, creative-framework",
        "use_cases": ["hero-section", "explainer-opener", "premium-design-asset"]
    },
    "CLAUDE STATIC AUTHORITY": {
        "category": "authority-visual",
        "brand_context": "executive, leadership, enterprise",
        "use_cases": ["authority-section", "trust-builder", "credibility-marker"]
    },
    "CLAUDE STATIC CREDIT CARD": {
        "category": "fintech-product",
        "brand_context": "premium, fintech, wealth, trust",
        "use_cases": ["product-showcase", "conversion-scene", "premium-indicator"]
    },
    "CLAUDE STATIC DIAMONDLUXUR": {
        "category": "luxury-asset",
        "brand_context": "premium, luxury, exclusivity, high-ticket",
        "use_cases": ["hero-section", "prestige-marker", "aspiration-trigger"]
    },
    "CLAUDE STATIC FOLDERDOCUMENTS": {
        "category": "knowledge-system",
        "brand_context": "enterprise, organization, authority, proof",
        "use_cases": ["proof-section", "credibility-anchor", "education-opener"]
    },
    "CLAUDE STATIC HAND": {
        "category": "human-gesture",
        "brand_context": "control, authority, action, leadership",
        "use_cases": ["hero-anchor", "CTA-focus", "authority-scene"]
    },
    "CLAUDE STATIC KNOWLEDGE": {
        "category": "intelligence-visual",
        "brand_context": "expertise, clarity, thought-leadership, premium-content",
        "use_cases": ["explainer-opener", "authority-section", "education-scene"]
    },
    "CLAUDE STATIC LAPTOP IMAGES, WORK STATION": {
        "category": "workspace-tech",
        "brand_context": "creator-economy, productivity, premium-workspace, enterprise",
        "use_cases": ["before-after", "productivity-proof", "workspace-hero"]
    },
    "CLAUDE STATIC MONEY ASSETS": {
        "category": "wealth-visual",
        "brand_context": "fintech, wealth, abundance, premium, high-ticket",
        "use_cases": ["conversion-trigger", "hero-section", "proof-section"]
    },
    "CLAUDE STATIC PHILOSOPHERS": {
        "category": "intellectual-visual",
        "brand_context": "thought-leadership, premium-content, expertise, authority",
        "use_cases": ["educational-opener", "authority-anchor", "premium-branding"]
    },
    "CLAUDE STATIC PORTRAIT": {
        "category": "human-portrait",
        "brand_context": "trust, authority, personal-brand, premium",
        "use_cases": ["hero-section", "authority-anchor", "credibility-visual"]
    },
    "CLAUDE STATIC STRATEGY": {
        "category": "strategic-visual",
        "brand_context": "premium, executive, enterprise, high-ticket",
        "use_cases": ["authority-section", "strategy-opener", "premium-explainer"]
    },
    "CLAUDE STATIC WATCHTIME ASSETS": {
        "category": "timevalue-visual",
        "brand_context": "urgency, premium, exclusive, creator-economy",
        "use_cases": ["CTA-scene", "scarcity-marker", "urgency-trigger"]
    }
}

def get_image_dimensions(img_path):
    """Extract image dimensions safely"""
    try:
        img = Image.open(img_path)
        return img.size
    except:
        return (0, 0)

def analyze_image_content(img_path):
    """Analyze image to detect objects and visual characteristics"""
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        
        if len(img_array.shape) < 2:
            return [], {}
        
        height, width = img_array.shape[:2]
        detected_objects = []
        confidence_scores = {}
        
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # Calculate pixel ratios
            skin_pixels = np.sum((img_array[:,:,0] > 90) & (img_array[:,:,0] < 220) & 
                                (img_array[:,:,1] > 40) & (img_array[:,:,1] < 180) &
                                (img_array[:,:,2] > 20) & (img_array[:,:,2] < 170) &
                                (img_array[:,:,0] > img_array[:,:,2]))
            skin_ratio = skin_pixels / (height * width) if (height * width) > 0 else 0
            
            light_pixels = np.sum((img_array[:,:,0] > 200) & (img_array[:,:,1] > 200) & (img_array[:,:,2] > 200))
            light_ratio = light_pixels / (height * width) if (height * width) > 0 else 0
            
            gold_pixels = np.sum((img_array[:,:,0] > 150) & (img_array[:,:,1] > 100) & (img_array[:,:,1] < 200) & (img_array[:,:,2] < 100))
            gold_ratio = gold_pixels / (height * width) if (height * width) > 0 else 0
            
            green_pixels = np.sum((img_array[:,:,1] > img_array[:,:,0]) & (img_array[:,:,1] > img_array[:,:,2]))
            green_ratio = green_pixels / (height * width) if (height * width) > 0 else 0
            
            blue_pixels = np.sum((img_array[:,:,2] > img_array[:,:,0]) & (img_array[:,:,2] > img_array[:,:,1]))
            blue_ratio = blue_pixels / (height * width) if (height * width) > 0 else 0
            
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            # Object detection
            if skin_ratio > 0.10:
                detected_objects.append("hand")
                confidence_scores["hand"] = min(0.95, skin_ratio * 2)
            
            screen_like = np.sum((img_array[:,:,0] > 80) & (img_array[:,:,0] < 200) & 
                                (img_array[:,:,1] > 80) & (img_array[:,:,1] < 200) &
                                (img_array[:,:,2] > 80) & (img_array[:,:,2] < 200))
            screen_ratio = screen_like / (height * width) if (height * width) > 0 else 0
            
            if contrast > 30 and (light_ratio > 0.10 or screen_ratio > 0.15):
                detected_objects.append("phone")
                detected_objects.append("mobile-device")
                confidence_scores["phone"] = min(0.85, 0.3 + (contrast / 255))
            
            if (light_ratio > 0.15 or screen_ratio > 0.20) and blue_ratio > 0.05:
                detected_objects.append("laptop")
                detected_objects.append("computer")
                confidence_scores["laptop"] = 0.75
            
            if gold_ratio > 0.12 and contrast > 40 and (width < height * 1.5):
                detected_objects.append("credit-card")
                detected_objects.append("payment-card")
                confidence_scores["credit-card"] = 0.70
            
            if light_ratio > 0.25 and contrast < 50:
                detected_objects.append("document")
                detected_objects.append("paper")
                confidence_scores["document"] = 0.75
            
            if (green_ratio > 0.15 or (gold_ratio > 0.12 and brightness < 180)):
                detected_objects.append("money")
                detected_objects.append("currency")
                confidence_scores["money"] = 0.70
            
            if contrast > 80 and brightness > 90 and brightness < 210:
                detected_objects.append("abstract-visual")
                confidence_scores["abstract-visual"] = 0.60
            
            if skin_ratio > 0.25:
                detected_objects.append("portrait")
                detected_objects.append("human-face")
                confidence_scores["portrait"] = 0.80
            
            if brightness < 120 and contrast > 50:
                detected_objects.append("dark-tone")
                confidence_scores["dark-tone"] = 0.65
        
        return list(set(detected_objects)), confidence_scores
    except Exception as e:
        return [], {}

def refine_detected_objects(detected_objects, folder_name, confidence_scores):
    """Apply folder-context refinement to detected objects"""
    refined = list(detected_objects)
    folder_lower = folder_name.lower()
    
    # HAND FOLDER: if we're IN the hand folder, hand is implied/present
    if "hand" in folder_lower:
        if "hand" not in refined:
            refined.append("hand")
            confidence_scores["hand"] = 0.70  # High confidence due to folder context
        if "phone" not in refined:
            refined.append("phone")  # Hand typically interacts with device
            confidence_scores["phone"] = 0.65
    
    # Credit card folder refinement
    if "credit card" in folder_lower and "dark-tone" in refined and "credit-card" not in refined:
        refined.append("credit-card")
        confidence_scores["credit-card"] = 0.75
    
    # Laptop/workstation refinement
    if ("laptop" in folder_lower or "work station" in folder_lower) and "laptop" not in refined:
        refined.append("laptop")
        confidence_scores["laptop"] = 0.70
    
    # Money refinement
    if "money" in folder_lower and "money" not in refined and "currency" not in refined:
        refined.append("money")
        confidence_scores["money"] = 0.70
    
    # Document refinement
    if ("folder" in folder_lower or "document" in folder_lower) and "document" not in refined:
        refined.append("document")
        confidence_scores["document"] = 0.70
    
    # Authority folder
    if "authority" in folder_lower and "authority" not in refined:
        refined.append("authority")
        confidence_scores["authority"] = 0.65
    
    # Portrait folder
    if "portrait" in folder_lower:
        if "portrait" not in refined:
            refined.append("portrait")
            confidence_scores["portrait"] = 0.75
        if "human-face" not in refined:
            refined.append("human-face")
            confidence_scores["human-face"] = 0.75
    
    return refined, confidence_scores

def build_tags_for_objects(detected_objects):
    """Build comprehensive tags based on detected objects"""
    tags = {
        "literal_tags": [],
        "symbolic_tags": [],
        "narrative_tags": [],
        "brand_tags": [],
        "motion_tags": [],
        "conversion_tags": []
    }
    
    # PHONE/MOBILE DEVICE
    if "phone" in detected_objects or "mobile-device" in detected_objects:
        tags["literal_tags"].extend(["phone", "mobile-device", "device", "screen", "technology", "smartphone"])
        tags["symbolic_tags"].extend(["communication", "connectivity", "connection", "interaction", 
                                     "instant-access", "modern-life", "always-on", "accessibility", "contact"])
        tags["narrative_tags"].extend(["communication-moment", "connection-anchor", "device-focus", 
                                      "interaction-scene", "digital-engagement", "call-to-action"])
        tags["brand_tags"].extend(["mobile-first", "connected", "modern", "tech-forward", "accessible", "immediate"])
        tags["motion_tags"].extend(["parallax-ready", "screen-glare", "tap-animation", "swipe-ready", "touch-reactive"])
        tags["conversion_tags"].extend(["mobile-interaction", "app-showcase", "communication-scene", 
                                       "connectivity-proof", "cta-ready"])
    
    # HAND
    if "hand" in detected_objects:
        tags["literal_tags"].extend(["hand", "gesture", "arm", "human-form", "body-part", "fingers", "palm"])
        tags["symbolic_tags"].extend(["action", "engagement", "control", "interaction", "connection",
                                     "human-touch", "personal-agency", "active-participation", "authority"])
        tags["narrative_tags"].extend(["action-anchor", "engagement-marker", "personal-connection", "call-to-action"])
        tags["brand_tags"].extend(["human-centric", "interactive", "personal", "accessible", "action-oriented"])
        tags["motion_tags"].extend(["gesture-emphasis", "touch-animation", "scale-readiness", "hand-gesture"])
        tags["conversion_tags"].extend(["call-to-action", "engagement-trigger", "human-moment", "activation-point"])
    
    # LAPTOP/COMPUTER
    if "laptop" in detected_objects or "computer" in detected_objects:
        tags["literal_tags"].extend(["laptop", "computer", "screen", "workspace", "monitor", "device", "tech"])
        tags["symbolic_tags"].extend(["productivity", "creation", "professional-work", "capability",
                                     "execution", "modern-work", "expertise-in-action", "capability-display"])
        tags["narrative_tags"].extend(["productivity-anchor", "work-in-progress", "capability-demo", "workspace-display"])
        tags["brand_tags"].extend(["productivity", "professional", "tech-enabled", "modern-workspace", "capable"])
        tags["motion_tags"].extend(["parallax-friendly", "screen-content-mask", "depth-effect", "screen-focus"])
        tags["conversion_tags"].extend(["productivity-proof", "capability-scene", "work-example", "professional-demo"])
    
    # CREDIT CARD
    if "credit-card" in detected_objects or "payment-card" in detected_objects:
        tags["literal_tags"].extend(["credit-card", "payment-card", "financial-product", "card", "plastic"])
        tags["symbolic_tags"].extend(["financial-power", "wealth-access", "premium-privilege", 
                                     "trust", "security", "opportunity", "status-symbol", "premium-access"])
        tags["narrative_tags"].extend(["wealth-marker", "premium-indicator", "product-focus", "premium-offering"])
        tags["brand_tags"].extend(["fintech", "premium", "high-ticket", "exclusive", "wealth", "elite"])
        tags["motion_tags"].extend(["flip-animation", "rotate-reveal", "card-presentation", "scale-emphasize"])
        tags["conversion_tags"].extend(["product-showcase", "premium-offering", "conversion-focus", "premium-landing"])
    
    # DOCUMENT/PAPER
    if "document" in detected_objects or "paper" in detected_objects:
        tags["literal_tags"].extend(["document", "paper", "page", "content", "information", "file"])
        tags["symbolic_tags"].extend(["authority", "proof", "credibility", "organization", "evidence",
                                     "knowledge", "transparency", "structure", "verification"])
        tags["narrative_tags"].extend(["proof-anchor", "credibility-marker", "evidence-display", "authority-foundation"])
        tags["brand_tags"].extend(["professional", "trustworthy", "organized", "transparent", "reliable"])
        tags["motion_tags"].extend(["page-turn", "reveal-animation", "document-showcase", "stack-cascade"])
        tags["conversion_tags"].extend(["proof-section", "credibility-anchor", "evidence-scene", "verification-display"])
    
    # MONEY/CURRENCY
    if "money" in detected_objects or "currency" in detected_objects:
        tags["literal_tags"].extend(["money", "currency", "cash", "wealth", "financial-asset", "payment"])
        tags["symbolic_tags"].extend(["abundance", "wealth", "financial-success", "opportunity",
                                     "ROI", "value", "investment", "prosperity", "freedom", "success"])
        tags["narrative_tags"].extend(["wealth-symbol", "opportunity-marker", "ROI-proof", "success-indicator"])
        tags["brand_tags"].extend(["wealth-focused", "fintech", "high-ticket", "prosperity", "premium"])
        tags["motion_tags"].extend(["floating-animation", "scale-emphasis", "glow-shine", "abundance-motion"])
        tags["conversion_tags"].extend(["conversion-trigger", "ROI-proof", "opportunity-scene", "success-marker"])
    
    # PORTRAIT/FACE
    if "portrait" in detected_objects or "human-face" in detected_objects:
        tags["literal_tags"].extend(["portrait", "face", "person", "expression", "human", "headshot"])
        tags["symbolic_tags"].extend(["trust", "credibility", "authenticity", "personal-connection",
                                     "leadership", "approachability", "confidence", "presence"])
        tags["narrative_tags"].extend(["trust-anchor", "personal-connection", "credibility-visual", "authentic-moment"])
        tags["brand_tags"].extend(["personal-brand", "trustworthy", "approachable", "professional", "authentic"])
        tags["motion_tags"].extend(["focus-zoom", "subtle-parallax", "portrait-spotlight", "fade-emphasis"])
        tags["conversion_tags"].extend(["trust-builder", "personal-branding", "credibility-scene", "authentic-connection"])
    
    # ABSTRACT/COMPLEX VISUALS
    if "abstract-visual" in detected_objects:
        tags["literal_tags"].extend(["abstract", "pattern", "complexity", "visual-texture", "design"])
        tags["symbolic_tags"].extend(["innovation", "complexity-mastered", "sophisticated-thinking", "creativity", "potential"])
        tags["narrative_tags"].extend(["creative-foundation", "concept-depth", "design-forward"])
        tags["brand_tags"].extend(["premium", "sophisticated", "modern", "innovative", "creative"])
        tags["motion_tags"].extend(["parallax-complex", "layered-animation", "organic-motion"])
        tags["conversion_tags"].extend(["creative-opener", "design-showcase", "innovation-marker"])
    
    # DARK TONE (authority, premium)
    if "dark-tone" in detected_objects:
        tags["symbolic_tags"].extend(["authority", "premium", "executive", "strategic", "power", "sophistication"])
        tags["brand_tags"].extend(["premium", "executive", "elite", "powerful", "sophisticated"])
    
    # AUTHORITY visual
    if "authority" in detected_objects:
        tags["literal_tags"].extend(["authority", "leadership", "command"])
        tags["symbolic_tags"].extend(["authority", "leadership", "power", "confidence", "control"])
        tags["narrative_tags"].extend(["authority-marker", "leadership-display"])
        tags["brand_tags"].extend(["premium", "executive", "leadership"])
        tags["motion_tags"].extend(["emphasis-zoom"])
        tags["conversion_tags"].extend(["authority-section"])
    
    # Remove duplicates while preserving order
    for key in tags:
        tags[key] = list(dict.fromkeys(tags[key]))
    
    return tags

def merge_tags(object_tags, folder_tags):
    """Merge object and folder tags intelligently"""
    merged = {}
    for key in object_tags:
        merged[key] = list(dict.fromkeys(object_tags[key] + folder_tags.get(key, [])))
    return merged

def get_folder_baseline_tags(folder_name):
    """Get folder-based baseline tags"""
    folder_lower = folder_name.lower()
    
    if "abstract" in folder_lower:
        return {
            "literal_tags": ["abstract", "shape", "composition"],
            "symbolic_tags": ["creativity", "innovation", "design"],
            "narrative_tags": ["design-foundation"],
            "brand_tags": ["premium", "modern"],
            "motion_tags": ["parallax-ready"],
            "conversion_tags": ["hero-section"]
        }
    elif "authority" in folder_lower:
        return {
            "literal_tags": ["professional", "executive"],
            "symbolic_tags": ["authority", "leadership", "confidence"],
            "narrative_tags": ["authority-anchor"],
            "brand_tags": ["premium", "executive"],
            "motion_tags": ["parallax-friendly"],
            "conversion_tags": ["authority-section"]
        }
    elif "hand" in folder_lower:
        return {
            "literal_tags": ["gesture", "human"],
            "symbolic_tags": ["action", "control"],
            "narrative_tags": ["action-marker"],
            "brand_tags": ["interactive"],
            "motion_tags": ["gesture-emphasis"],
            "conversion_tags": ["cta-scene"]
        }
    elif "portrait" in folder_lower:
        return {
            "literal_tags": ["person", "expression"],
            "symbolic_tags": ["trust", "authenticity"],
            "narrative_tags": ["personal-anchor"],
            "brand_tags": ["personal-brand"],
            "motion_tags": ["portrait-focus"],
            "conversion_tags": ["trust-scene"]
        }
    else:
        return {
            "literal_tags": ["asset"],
            "symbolic_tags": ["professional"],
            "narrative_tags": ["general-use"],
            "brand_tags": ["professional"],
            "motion_tags": ["fade-compatible"],
            "conversion_tags": ["general-use"]
        }

def process_all_images(root_dir):
    """Process all images with content analysis and folder refinement"""
    metadata_collection = []
    folder_stats = defaultdict(lambda: {"count": 0, "errors": 0})
    all_objects = defaultdict(int)
    
    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        
        if not os.path.isdir(folder_path) or folder_name.startswith('.'):
            continue
        
        context = FOLDER_CONTEXTS.get(folder_name, {})
        
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                img_path = os.path.join(folder_path, filename)
                
                try:
                    width, height = get_image_dimensions(img_path)
                    file_size = os.path.getsize(img_path)
                    
                    # Analyze image content
                    detected_objects, confidence_scores = analyze_image_content(img_path)
                    
                    # Refine with folder context
                    detected_objects, confidence_scores = refine_detected_objects(
                        detected_objects, folder_name, confidence_scores
                    )
                    
                    # Track objects
                    for obj in detected_objects:
                        all_objects[obj] += 1
                    
                    # Generate file hash
                    with open(img_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    
                    # Build object tags + folder baseline
                    object_tags = build_tags_for_objects(detected_objects)
                    folder_tags = get_folder_baseline_tags(folder_name)
                    tags = merge_tags(object_tags, folder_tags)
                    
                    # Create metadata entry
                    metadata_entry = {
                        "id": file_hash,
                        "filename": filename,
                        "folder": folder_name,
                        "relative_path": os.path.relpath(img_path, root_dir),
                        "file_size_bytes": file_size,
                        "dimensions": {
                            "width": width,
                            "height": height,
                            "aspect_ratio": f"{width}:{height}" if height > 0 else "unknown"
                        },
                        "detected_objects": detected_objects,
                        "object_confidence": confidence_scores,
                        "folder_context": context,
                        "literal_tags": tags["literal_tags"],
                        "symbolic_tags": tags["symbolic_tags"],
                        "narrative_tags": tags["narrative_tags"],
                        "brand_tags": tags["brand_tags"],
                        "motion_tags": tags["motion_tags"],
                        "conversion_tags": tags["conversion_tags"],
                        "tag_count": sum(len(v) for k, v in tags.items() if k.endswith("_tags")),
                        "confidence_notes": f"Tags derived from detected objects: {detected_objects}. Combined with folder context for semantic depth."
                    }
                    
                    metadata_collection.append(metadata_entry)
                    folder_stats[folder_name]["count"] += 1
                    
                except Exception as e:
                    folder_stats[folder_name]["errors"] += 1
                    print(f"⚠️  Error: {filename} - {str(e)[:50]}")
    
    return metadata_collection, dict(folder_stats), dict(all_objects)

def main():
    root_dir = r"c:\Users\HomePC\Downloads\MATT MANHUNTER"
    
    print("🏗️  PROMETHEUS Asset Tagging Engine v3 - CONTENT-AWARE + REFINED")
    print("=" * 70)
    print(f"Processing: {root_dir}")
    print()
    
    metadata_collection, stats, all_objects = process_all_images(root_dir)
    
    print(f"✅ Total images processed: {len(metadata_collection)}")
    print()
    print("Folder breakdown:")
    for folder in sorted(stats.keys()):
        s = stats[folder]
        print(f"  • {folder}: {s['count']} images", end="")
        if s['errors'] > 0:
            print(f" ({s['errors']} errors)", end="")
        print()
    
    output_file = os.path.join(root_dir, "PROMETHEUS_ASSET_METADATA.json")
    output_data = {
        "metadata_version": "3.0",
        "total_assets": len(metadata_collection),
        "tagging_method": "Content-Aware Object Detection + Folder Context Refinement",
        "tagging_schema": {
            "layers": ["literal_tags", "symbolic_tags", "narrative_tags", "brand_tags", "motion_tags", "conversion_tags"],
            "min_tags_per_layer": 5,
            "recommended_total": "20-40+ tags per image"
        },
        "assets": metadata_collection
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"✅ Saved: {output_file}")
    print()
    
    total_tags = sum(e['tag_count'] for e in metadata_collection)
    avg_tags = total_tags / len(metadata_collection) if metadata_collection else 0
    
    print("📊 Statistics:")
    print(f"  • Total images: {len(metadata_collection)}")
    print(f"  • Total tags: {total_tags}")
    print(f"  • Avg tags/image: {avg_tags:.1f}")
    print()
    print("🔍 Detected Objects:")
    for obj in sorted(all_objects.keys(), key=lambda x: all_objects[x], reverse=True):
        print(f"  • {obj}: {all_objects[obj]}")
    print()
    print("✨ Assets ready for revenue-generating composition systems!")

if __name__ == "__main__":
    main()
