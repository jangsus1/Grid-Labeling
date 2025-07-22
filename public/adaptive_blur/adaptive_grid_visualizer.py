import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import json
from collections import defaultdict
from ortools.sat.python import cp_model


def rect_to_grid(rectangles, n_rows, n_cols):
    """Convert a list of rectangle coordinates to a binary grid."""
    area = np.zeros((n_rows, n_cols), dtype=np.int32)
    for r, c in rectangles:
        area[r, c] = 1
    return area

def compute_integral_image(area):
    """ Computes the integral image for quick sum queries. """
    return np.cumsum(np.cumsum(area, axis=0), axis=1)

def is_tile_valid(area, integral, w, h):
    """ Check if a w x h tile can fit into the area using the integral image. """
    rows, cols = area.shape
    for i in range(rows - h + 1):
        for j in range(cols - w + 1):
            # Compute the sum of the w x h submatrix using the integral image
            total = integral[i + h - 1, j + w - 1]
            if i > 0:
                total -= integral[i - 1, j + w - 1]
            if j > 0:
                total -= integral[i + h - 1, j - 1]
            if i > 0 and j > 0:
                total += integral[i - 1, j - 1]
            # If the entire subgrid is filled with 1s, the tile is valid
            if total == w * h:
                return True
    return False

def get_valid_tiles(area, tile_shape, max_tile_size):
    """ Generate all valid tiles that fit within the binary grid. """
    area = np.array(area)  # Ensure it's a NumPy array
    rows, cols = area.shape
    
    # Find bounding box of occupied region for efficiency
    occupied_rows, occupied_cols = np.where(area == 1)
    if len(occupied_rows) == 0:
        return []
    
    min_row, max_row = occupied_rows.min(), occupied_rows.max()
    min_col, max_col = occupied_cols.min(), occupied_cols.max()
    
    # Limit max tile size to the bounding box dimensions
    max_possible_width = max_col - min_col + 1
    max_possible_height = max_row - min_row + 1
    effective_max_tile = min(max_tile_size, max(max_possible_width, max_possible_height))
    
    integral = compute_integral_image(area)
    tiles = []
    
    if tile_shape == "square":
        for s in range(1, effective_max_tile + 1):
            if s <= max_possible_width and s <= max_possible_height:
                if is_tile_valid(area, integral, s, s):
                    tiles.append((s, s))
    else:  # rectangle
        for w in range(1, min(effective_max_tile, max_possible_width) + 1):
            for h in range(1, min(effective_max_tile, max_possible_height) + 1):
                if is_tile_valid(area, integral, w, h):
                    tiles.append((w, h))
    
    if len(tiles) == 0:
        # Fallback: at least 1x1 tiles should work if there are occupied cells
        tiles = [(1, 1)]
    
    return tiles

def optimal_tiling(rectangles, n_rows, n_cols, tile_shape="square", max_tile=4, aspect_ratio_penalty=0.01):
    """Solve the minimum tile cover problem using ILP with OR-Tools, with optional aspect ratio penalty."""
    if len(rectangles) == 0:
        return []
        
    model = cp_model.CpModel()
    
    # Convert rectangles to a binary grid
    grid = rect_to_grid(rectangles, n_rows, n_cols)
    
    # Adjust max tile size
    if max_tile == "max":
        max_tile = max(n_rows, n_cols)
    
    # Get valid tile sizes efficiently - only for tiles that can actually fit in this specific region
    valid_tiles = get_valid_tiles(grid, tile_shape, max_tile)
    
    print(f"    [ILP] Optimizing {len(rectangles)} cells with {len(valid_tiles)} valid tile sizes...")
    
    # Define ILP Decision Variables more efficiently
    tile_vars = {}
    occupied_cells = set(rectangles)  # Convert to set for O(1) lookup
    
    # Only create variables for positions where tiles can actually be placed
    for i in range(n_rows):
        for j in range(n_cols):
            for w, h in valid_tiles:
                # Check if tile placement is valid (within bounds and all cells occupied)
                if i + h <= n_rows and j + w <= n_cols:
                    # Check if ALL cells that this tile would cover are occupied
                    tile_valid = True
                    for di in range(h):
                        for dj in range(w):
                            if (i + di, j + dj) not in occupied_cells:
                                tile_valid = False
                                break
                        if not tile_valid:
                            break
                    
                    if tile_valid:
                        tile_vars[(i, j, w, h)] = model.NewBoolVar(f't_{i}_{j}_{w}_{h}')

    print(f"    [ILP] Created {len(tile_vars)} decision variables")
    
    if len(tile_vars) == 0:
        print("    [ILP] No valid tile placements found!")
        return []

    # Constraint 1: Cover every occupied cell exactly once (combined coverage + non-overlap)
    for r, c in rectangles:
        covering_tiles = []
        for (i, j, w, h), var in tile_vars.items():
            # Check if this tile covers cell (r, c)
            if i <= r < i + h and j <= c < j + w:
                covering_tiles.append(var)
        
        if len(covering_tiles) == 0:
            print(f"    [ILP] Warning: No tiles can cover cell ({r}, {c})")
            return []  # Infeasible
        
        # Each cell must be covered by exactly one tile
        model.Add(sum(covering_tiles) == 1)

    print(f"    [ILP] Added {len(rectangles)} coverage constraints")

    # Objective: Minimize the number of tiles, and penalize aspect ratio
    objective_terms = []
    for (i, j, w, h), var in tile_vars.items():
        aspect_penalty = abs(w - h)
        # Add 1 for each tile, plus aspect penalty
        objective_terms.append(var)
        if aspect_ratio_penalty > 0:
            # Add penalty for non-square tiles
            objective_terms.append(aspect_penalty * aspect_ratio_penalty * var)
    model.Minimize(sum(objective_terms))

    # Solve ILP Model with improved parameters
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120  # Increased timeout for complex problems
    solver.parameters.num_search_workers = 4    # Use parallel search
    solver.parameters.log_search_progress = False  # Reduce output noise
    
    print(f"    [ILP] Solving...")
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL:
        print(f"    [ILP] Found optimal solution with {solver.ObjectiveValue()} tiles (with aspect penalty)")
    elif status == cp_model.FEASIBLE:
        print(f"    [ILP] Found feasible solution with {solver.ObjectiveValue()} tiles (timeout, with aspect penalty)")
    elif status == cp_model.INFEASIBLE:
        print("    [ILP] Problem is infeasible")
        return []
    else:
        print(f"    [ILP] Solver failed with status: {status}")
        return []

    # Extract solution
    solution = []
    for (i, j, w, h), var in tile_vars.items():
        if solver.Value(var) == 1:
            solution.append((i, j, w, h))

    print(f"    [ILP] Extracted {len(solution)} tiles from solution")
    return solution

def ocr_stub(image_path, ocr_dict):
    """Stub OCR function - replace with your actual OCR implementation"""
    basename = os.path.basename(image_path)
    img = cv2.imread(image_path)
    text_areas = []
    
    if basename in ocr_dict:
        for bbox in ocr_dict[basename]:
            left, top = bbox[0]
            right, bottom = bbox[2]
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)

            left = max(0, min(left, img.shape[1] - 1))
            right = max(0, min(right, img.shape[1] - 1))
            top = max(0, min(top, img.shape[0] - 1))
            bottom = max(0, min(bottom, img.shape[0] - 1))

            top_line = img[top, left:right, :].reshape(-1, 3)
            bottom_line = img[bottom, left:right, :].reshape(-1, 3)
            left_line = img[top:bottom, left, :].reshape(-1, 3)
            right_line = img[top:bottom, right, :].reshape(-1, 3)
            lines = np.vstack([top_line, bottom_line, left_line, right_line])
            R_color = np.argmax(np.bincount(lines[:, 0]))
            G_color = np.argmax(np.bincount(lines[:, 1]))
            B_color = np.argmax(np.bincount(lines[:, 2]))
            R_color, G_color, B_color = int(R_color), int(G_color), int(B_color)
            # filled rectangle
            cv2.rectangle(img, (left, top), (right, bottom),
                          (R_color, G_color, B_color), -1)
            ## y, x, h, w
            text_areas.append([top, left, bottom - top, right - left])
    return img, text_areas

def canny(img):
    """Apply Canny edge detection to image"""
    channels = cv2.split(img)
    # Apply Canny to each channel
    edges = [cv2.Canny(channel, 50, 150) for channel in channels]
    # Combine edges using bitwise OR
    combined_edges = cv2.bitwise_or(edges[0], edges[1])
    combined_edges = cv2.bitwise_or(combined_edges, edges[2])
    return combined_edges

def optimized_tile(edges, text_area, min_grid_ratio, tile_shape="rectangle", max_tile="max"):
    """Main tiling optimization function"""
    min_x = int(min_grid_ratio * edges.shape[1])
    min_y = int(min_grid_ratio * edges.shape[0])

    # Calculate how many cells fit (rows x columns)
    n_rows = (edges.shape[0] + min_y - 1) // min_y
    n_cols = (edges.shape[1] + min_x - 1) // min_x
    
    text_grid = np.zeros((n_rows, n_cols), dtype=np.int32)
    text_rectangles = []
    for i, (y_text, x_text, h_text, w_text) in enumerate(text_area):
        subrects = []
        for row in range(n_rows):
            for col in range(n_cols):
                if text_grid[row, col] > 0: continue
                y_grid = row * min_y
                x_grid = col * min_x
                h_grid = min_y
                w_grid = min_x
                # if two box overlap
                if (x_text < x_grid + w_grid and x_text + w_text > x_grid and
                    y_text < y_grid + h_grid and y_text + h_text > y_grid):
                    text_grid[row, col] = i+1
                    subrects.append((row, col))
                    
        text_rectangles.append(subrects)
    
    has_text_grid = text_grid > 0
    has_edge_grid = np.zeros((n_rows, n_cols), dtype=bool)
    

    # For each cell in the grid, check if it has any edge pixels
    for row in range(n_rows):
        for col in range(n_cols):
            if has_text_grid[row, col]:
                has_edge_grid[row, col] = False
                continue
            y_start = row * min_y
            x_start = col * min_x
            h = min_y
            w = min_x
            # Crop cell
            cell = edges[y_start:y_start+h, x_start:x_start+w]
            has_edge_grid[row, col] = cell.any()
    
    
    # assert no overlap btw text and edge
    assert np.all(~(has_text_grid & has_edge_grid))

    # Find connected components in the "no-edge" cells
    noedge_num_labels, noedge_labels = cv2.connectedComponents((~(has_edge_grid | has_text_grid)).astype(np.uint8))

    background_rectangles_groups = []
    for label_id in range(1, noedge_num_labels):
        points = np.argwhere(noedge_labels == label_id)
        if len(points) == 0:
            continue
        group_subrects = []
        for r, c in points:
            group_subrects.append((r, c))
        background_rectangles_groups.append(group_subrects)

    edge_num_labels, edge_labels = cv2.connectedComponents(has_edge_grid.astype(np.uint8))
    edge_rectangles_groups = []
    for label_id in range(1, edge_num_labels):
        points = np.argwhere(edge_labels == label_id)
        if len(points) == 0:
            continue
        group_subrects = []
        for r, c in points:
            group_subrects.append((r, c))
        edge_rectangles_groups.append(group_subrects)

    # STEP 1: Merge background rectangles among themselves
    print(f"  [TILING] Processing {len(background_rectangles_groups)} background connected components...")
    background_solutions = []
    for i, rectangles in enumerate(background_rectangles_groups):
        if len(rectangles) == 0:continue
        print(f"    [BACKGROUND] Component {i+1}: {len(rectangles)} cells")
        solution = optimal_tiling(rectangles, n_rows, n_cols, tile_shape, max_tile)
        background_solutions.extend([(r*min_y, c*min_x, w*min_x, h*min_y) for r, c, w, h in solution])
    
    print(f"  [TILING] Processing {len(edge_rectangles_groups)} edge connected components...")
    edge_solutions = []
    for i, rectangles in enumerate(edge_rectangles_groups):
        if len(rectangles) == 0:continue
        print(f"    [EDGE] Component {i+1}: {len(rectangles)} cells")
        solution = optimal_tiling(rectangles, n_rows, n_cols, tile_shape, max_tile)
        edge_solutions.extend([(r*min_y, c*min_x, w*min_x, h*min_y) for r, c, w, h in solution])
    
    print(f"  [TILING] Processing {len(text_rectangles)} text regions...")
    text_solutions = []
    for i, rectangles in enumerate(text_rectangles):
        if len(rectangles) == 0:continue
        print(f"    [TEXT] Region {i+1}: {len(rectangles)} cells")
        solution = optimal_tiling(rectangles, n_rows, n_cols, tile_shape, max_tile)
        text_solutions.extend([(r*min_y, c*min_x, w*min_x, h*min_y) for r, c, w, h in solution])
    
    print(f"  [TILING] Total tiles: Background={len(background_solutions)}, Edge={len(edge_solutions)}, Text={len(text_solutions)}")
    return background_solutions, edge_solutions, text_solutions


def visualize_adaptive_grid_steps(image_path, output_folder="adaptive_grid_visualization", 
                                min_grid_ratio=0.08, tile_shape="rect", max_tile=100,
                                text_color=(255, 100, 100), edge_color=(100, 255, 100), 
                                blank_color=(100, 100, 255), grid_color=(128, 128, 128),
                                ocr_dict=None):
    """
    Generate visualization images for each step of the adaptive grid algorithm.
    
    Parameters:
    - image_path: Path to input image
    - output_folder: Folder to save visualization images
    - min_grid_ratio: Grid ratio parameter for tiling
    - tile_shape: Shape of tiles ("rect" or "square")
    - max_tile: Maximum tile size
    - text_color, edge_color, blank_color: RGB colors for different regions
    - grid_color: RGB color for final grid lines
    - ocr_dict: Dictionary containing OCR results (if None, will create empty text areas)
    """
    
    print("[Step 1] Creating output folder and loading original image...")
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Load original image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not load image from {image_path}")
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    print("[Step 1] Original image loaded.")
    
    # Save original image
    plt.figure(figsize=(12, 8))
    plt.imshow(original_img_rgb)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '1_original.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Step 1] Original image saved.")
    
    # Step 2: OCR and text detection
    print("[Step 2] Running OCR and text detection...")
    if ocr_dict is None:
        ocr_dict = {}
    ocr_removed, text_areas = ocr_stub(image_path, ocr_dict)
    ocr_img_rgb = cv2.cvtColor(ocr_removed, cv2.COLOR_BGR2RGB)
    print(f"[Step 2] OCR and text detection complete. {len(text_areas)} text areas found.")
    
    # Visualize detected text areas
    plt.figure(figsize=(12, 8))
    plt.imshow(original_img_rgb)
    ax = plt.gca()
    
    for i, (y, x, h, w) in enumerate(text_areas):
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=np.array(text_color)/255, 
                               facecolor=np.array(text_color)/255, alpha=0.3)
        ax.add_patch(rect)
        plt.text(x + w/2, y + h/2, f'T{i+1}', ha='center', va='center', 
                fontsize=10, color='white', weight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '2_text_detection.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Step 2] Text detection visualization saved.")
    
    # Step 3: Edge detection
    print("[Step 3] Running edge detection (Canny)...")
    edges = canny(ocr_removed)
    print("[Step 3] Edge detection complete.")
    
    plt.figure(figsize=(12, 8))
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '3_edge_detection.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Step 3] Edge detection visualization saved.")
    
    # Step 4: Grid division and region classification
    print("[Step 4] Dividing grid and classifying regions...")
    min_x = int(min_grid_ratio * edges.shape[1])
    min_y = int(min_grid_ratio * edges.shape[0])
    n_rows = (edges.shape[0] + min_y - 1) // min_y
    n_cols = (edges.shape[1] + min_x - 1) // min_x
    
    # Create grids for text, edge, and blank regions
    text_grid = np.zeros((n_rows, n_cols), dtype=np.int32)
    has_edge_grid = np.zeros((n_rows, n_cols), dtype=bool)
    
    # Mark text regions
    for i, (y_text, x_text, h_text, w_text) in enumerate(text_areas):
        for row in range(n_rows):
            for col in range(n_cols):
                if text_grid[row, col] > 0: continue
                y_grid = row * min_y
                x_grid = col * min_x
                h_grid = min_y
                w_grid = min_x
                if (x_text < x_grid + w_grid and x_text + w_text > x_grid and
                    y_text < y_grid + h_grid and y_text + h_text > y_grid):
                    text_grid[row, col] = i+1
    
    has_text_grid = text_grid > 0
    
    # Mark edge regions
    for row in range(n_rows):
        for col in range(n_cols):
            if has_text_grid[row, col]:
                has_edge_grid[row, col] = False
                continue
            y_start = row * min_y
            x_start = col * min_x
            cell = edges[y_start:y_start+min_y, x_start:x_start+min_x]
            has_edge_grid[row, col] = cell.any()
    
    print("[Step 4] Grid division and region classification complete.")
    
    # Visualize grid regions
    plt.figure(figsize=(12, 8))
    plt.imshow(original_img_rgb)
    ax = plt.gca()
    
    # Generate aesthetic colors for different tile types
    np.random.seed(42)  # For consistent colors
    tile_colors = plt.cm.Set3(np.linspace(0, 1, 12))  # Use Set3 colormap for aesthetic colors
    
    for row in range(n_rows):
        for col in range(n_cols):
            y_start = row * min_y
            x_start = col * min_x
            
            if has_text_grid[row, col]:
                color = np.array(text_color) / 255
                label = 'Text'
            elif has_edge_grid[row, col]:
                color = np.array(edge_color) / 255
                label = 'Edge'
            else:
                color = np.array(blank_color) / 255
                label = 'Blank'
            
            rect = patches.Rectangle((x_start, y_start), min_x, min_y, 
                                   linewidth=1, edgecolor='white', alpha=0.4,
                                   facecolor=color)
            ax.add_patch(rect)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '4_grid_regions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Step 4] Grid region visualization saved.")
    
    # Step 5: Optimized tiling result
    print("[Step 5] Running optimized tiling (ILP)... This may take a while for large images.")
    background_solutions, edge_solutions, text_solutions = optimized_tile(
        edges, text_areas, min_grid_ratio, tile_shape, max_tile)
    print(f"[Step 5] Optimized tiling complete. Background: {len(background_solutions)}, Edge: {len(edge_solutions)}, Text: {len(text_solutions)} tiles.")
    
    plt.figure(figsize=(12, 8))
    plt.imshow(original_img_rgb)
    ax = plt.gca()
    
    # Draw optimized tiles with consistent colors for each type
    for i, (y, x, w, h) in enumerate(background_solutions):
        color = np.array(blank_color) / 255
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color, facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        plt.text(x + w/2, y + h/2, f'B{i+1}', ha='center', va='center', 
                fontsize=8, color='black', weight='bold')
    
    for i, (y, x, w, h) in enumerate(edge_solutions):
        color = np.array(edge_color) / 255
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color, facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        plt.text(x + w/2, y + h/2, f'E{i+1}', ha='center', va='center', 
                fontsize=8, color='black', weight='bold')
    
    for i, (y, x, w, h) in enumerate(text_solutions):
        color = np.array(text_color) / 255
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color, facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        plt.text(x + w/2, y + h/2, f'T{i+1}', ha='center', va='center', 
                fontsize=8, color='black', weight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '5_optimized_tiles.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Step 5] Optimized tiling visualization saved.")
    
    # Step 6: Final grid overlay
    print("[Step 6] Drawing final grid overlay...")
    rectangles = background_solutions + edge_solutions + text_solutions
    
    plt.figure(figsize=(12, 8))
    plt.imshow(original_img_rgb)
    ax = plt.gca()
    
    # Draw final grid lines
    for y, x, w, h in rectangles:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=np.array(grid_color)/255, 
                               facecolor='none')
        ax.add_patch(rect)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '6_final_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Step 6] Final grid overlay saved.")
    
    # Step 7: Summary statistics
    print("[Step 7] Saving summary statistics...")
    stats = {
        'total_tiles': len(rectangles),
        'background_tiles': len(background_solutions),
        'edge_tiles': len(edge_solutions),
        'text_tiles': len(text_solutions),
        'grid_ratio': min_grid_ratio,
        'image_size': original_img.shape[:2]
    }
    
    # Save statistics
    with open(os.path.join(output_folder, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Visualization completed! Images saved in '{output_folder}' folder.")
    print(f"Statistics: {stats}")
    
    return rectangles, stats


if __name__ == "__main__":
    print("[Main] Starting adaptive grid visualization script...")
    # Example usage - replace with your image path
    image_path = "/Users/minsukchang/Research/ChartDataset/chartqa/05411753006467.png"
    
    # Load OCR dictionary if available
    ocr_dict = {}
    try:
        with open("/Users/minsukchang/Research/ChartDataset/OCR/chartqa.json") as f:
            ocr_dict.update(json.load(f))
    except FileNotFoundError:
        print("OCR file not found, proceeding without text detection")
    
    # Generate visualizations
    rectangles, stats = visualize_adaptive_grid_steps(
        image_path=image_path,
        output_folder="adaptive_grid_demo",
        min_grid_ratio=0.05,
        tile_shape="rect",
        max_tile=400,
        text_color=(255, 100, 100),    # Red for text
        edge_color=(100, 255, 100),    # Green for edges
        blank_color=(100, 100, 255),   # Blue for blank areas
        grid_color=(64, 64, 64),       # Dark gray for final grid
        ocr_dict=ocr_dict
    ) 