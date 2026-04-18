"""
STEP 1: Simulated Optical Detection Layer
Mimics a YOLO-based space object detection system.
Adds realistic sensor noise to true SGP4 positions.
"""
import numpy as np

class YOLOSimulator:
    def __init__(self, noise_std=0.5):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise (km) added to simulate sensor uncertainty.
        """
        self.noise_std = noise_std

    def detect(self, objects):
        """
        Simulate detection of space objects from environment data.
        
        Args:
            objects: list of dicts with keys 'name', 'position' (np.array), 'type'
            
        Returns:
            list of detection dicts with observed (noisy) position and confidence score.
        """
        detections = []
        for obj in objects:
            true_pos = np.array(obj['position'], dtype=float)
            
            # Skip invalid positions
            if np.any(np.isnan(true_pos)):
                continue
            
            # Add Gaussian noise to simulate sensor measurement error
            noise = np.random.normal(0, self.noise_std, size=3)
            observed_pos = true_pos + noise
            
            # Confidence inversely proportional to noise magnitude
            noise_mag = np.linalg.norm(noise)
            confidence = max(0.0, min(1.0, 1.0 - (noise_mag / (3 * self.noise_std))))
            
            detections.append({
                'name': obj.get('name', 'UNKNOWN'),
                'observed_position': observed_pos,
                'true_position': true_pos,
                'confidence_score': round(confidence, 4),
                'type': obj.get('type', 'debris'),
                'bbox': self._generate_bbox(observed_pos)
            })
            
        return detections

    def _generate_bbox(self, position):
        """Generate a simulated bounding box around the detected object."""
        half_size = np.random.uniform(0.01, 0.05)
        return {
            'center': position.tolist(),
            'half_width': half_size
        }
