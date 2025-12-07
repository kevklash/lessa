"""
Feature Cache System for LESSA
Pre-processes and caches features for fast recognition performance.
"""

import numpy as np
import pickle
import json
import os
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

class FeatureCache:
    """Efficient feature storage and retrieval system."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize feature cache."""
        self.cache_dir = cache_dir
        self.features_file = os.path.join(cache_dir, "processed_features.pkl")
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache data
        self.cached_features = {}  # {letter: [feature_vectors]}
        self.cached_labels = []
        self.cache_metadata = {}
        
        print(f"🗃️  Feature cache initialized: {cache_dir}")
    
    def _get_data_hash(self, data_file: str) -> str:
        """Get hash of the data file to detect changes."""
        if not os.path.exists(data_file):
            return ""
        
        # Get file modification time and size for quick change detection
        stat = os.stat(data_file)
        hash_content = f"{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def _load_cache_metadata(self) -> Dict:
        """Load cache metadata."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_cache_metadata(self, metadata: Dict):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save cache metadata: {e}")
    
    def is_cache_valid(self, data_file: str) -> bool:
        """Check if cached features are up to date."""
        if not os.path.exists(self.features_file):
            return False
        
        metadata = self._load_cache_metadata()
        current_hash = self._get_data_hash(data_file)
        cached_hash = metadata.get('data_hash', '')
        
        return current_hash == cached_hash and current_hash != ""
    
    def build_cache(self, data_file: str, feature_extractor) -> bool:
        """Build feature cache from raw data file."""
        print(f"🔄 Building feature cache from {data_file}...")
        start_time = time.time()
        
        try:
            # Load raw data
            if not os.path.exists(data_file):
                print(f"❌ Data file not found: {data_file}")
                return False
            
            with open(data_file, 'r') as f:
                raw_data = json.load(f)
            
            # Extract and cache features
            cached_features = {}
            all_features = []
            all_labels = []
            total_samples = 0
            
            for letter, samples in raw_data.items():
                if len(samples) == 0:
                    continue
                
                letter_features = []
                
                for sample in samples:
                    try:
                        # Extract features using existing method
                        feature_vector = feature_extractor._extract_features_from_sample(sample)
                        
                        if feature_vector is not None and len(feature_vector) > 0:
                            letter_features.append(feature_vector)
                            all_features.append(feature_vector)
                            all_labels.append(letter)
                            total_samples += 1
                    except Exception as e:
                        print(f"⚠️  Error processing sample for {letter}: {e}")
                        continue
                
                if letter_features:
                    cached_features[letter] = np.array(letter_features)
            
            # Save cached features
            cache_data = {
                'features_by_letter': cached_features,
                'all_features': np.array(all_features) if all_features else np.array([]),
                'all_labels': np.array(all_labels) if all_labels else np.array([]),
                'feature_dimension': len(all_features[0]) if all_features else 0,
                'total_samples': total_samples
            }
            
            with open(self.features_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            metadata = {
                'data_hash': self._get_data_hash(data_file),
                'created_at': datetime.now().isoformat(),
                'total_samples': total_samples,
                'letters': list(cached_features.keys()),
                'feature_dimension': cache_data['feature_dimension']
            }
            self._save_cache_metadata(metadata)
            
            # Update instance cache
            self.cached_features = cached_features
            self.cached_labels = all_labels
            self.cache_metadata = metadata
            
            build_time = time.time() - start_time
            print(f"✅ Feature cache built successfully!")
            print(f"   • Processed {total_samples} samples")
            print(f"   • Letters: {', '.join(sorted(cached_features.keys()))}")
            print(f"   • Feature dimension: {cache_data['feature_dimension']}")
            print(f"   • Build time: {build_time:.2f}s")
            print(f"   • Cache size: {os.path.getsize(self.features_file) / 1024:.1f}KB")
            
            return True
            
        except Exception as e:
            print(f"❌ Error building feature cache: {e}")
            return False
    
    def load_cache(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """Load cached features for training."""
        if not os.path.exists(self.features_file):
            return None, None, {}
        
        try:
            with open(self.features_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.cached_features = cache_data.get('features_by_letter', {})
            self.cache_metadata = self._load_cache_metadata()
            
            X = cache_data.get('all_features')
            y = cache_data.get('all_labels')
            
            print(f"📦 Loaded cached features:")
            print(f"   • {len(X) if X is not None else 0} samples")
            print(f"   • {len(self.cached_features)} letters")
            print(f"   • Feature dimension: {cache_data.get('feature_dimension', 0)}")
            
            return X, y, self.cached_features
            
        except Exception as e:
            print(f"❌ Error loading feature cache: {e}")
            return None, None, {}
    
    def get_features_for_letter(self, letter: str) -> Optional[np.ndarray]:
        """Get all cached features for a specific letter."""
        return self.cached_features.get(letter)
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'cache_exists': os.path.exists(self.features_file),
            'cache_size_kb': os.path.getsize(self.features_file) / 1024 if os.path.exists(self.features_file) else 0,
            'metadata': self.cache_metadata
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            if os.path.exists(self.features_file):
                os.remove(self.features_file)
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
            
            self.cached_features = {}
            self.cached_labels = []
            self.cache_metadata = {}
            
            print("🗑️  Feature cache cleared")
            
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")