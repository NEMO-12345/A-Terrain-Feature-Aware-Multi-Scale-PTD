# A-Terrain-Feature-Aware-Multi-Scale-PTD
We propose a terrain feature-aware multi-scale PTD algorithm that enhances ground filtering accuracy in complex terrains through TIN-guided seed selection, adaptive boundary expansion, and terrain-specific compensation models

# Clone the repository
git clone https://github.com/NEMO-12345/A-Terrain-feature-Aware-multi-Scale-PTD.git
cd A-Terrain-feature-Aware-multi-Scale-PTD

# Install dependencies
pip install -r requirements.txt



# demo.py 
def main():
    # ========== PARAMETER CONFIGURATION ==========
    
    # Filter parameters (critical for performance)
    W_max = 40      
    S_min = 0.5     
    S_max = 1.2    
    Angle_min = 20   
    
    # File paths
    input_las_file = r"input_pointcloud.las"  # Input LAS file
    output_dir = r"results/"                   # Output directory
    final_output_las = r"ground_points.las"    # Output filename
    
    
    # ... rest of the code ...



  | Parameter | Range | Selection Criteria |
|-----------|-------|-------------------|
| W<sub>max</sub> | 10 - 40m | Control the initial grid resolution. Refer to the diagonal length of the largest building size; for mountainous areas without buildings and with significant terrain undulations, the parameters should be appropriately reduced. |
| θ<sub>thr</sub> | 15° - 88° | The angle criterion for distinguishing between ground points and non-ground points. Based on the maximum slope of the terrain in the sample area. |
| S<sub>max</sub> | 1.2 - 1.5m | The distance criterion for distinguishing between ground points and non-ground points in the bottom layer TIN. Refer to the average distance between non-ground points and ground points within the initial grid. |
| S<sub>min</sub> | 0.4 - 0.7m | The distance criterion for distinguishing between ground points and non-ground points in the top layer TIN. Refer to the minimum distance between non-ground points and ground points. |"""
