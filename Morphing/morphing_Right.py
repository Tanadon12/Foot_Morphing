import bpy
import mathutils
import bmesh

import math
from bpy import context

def get_object_dimensions(object_name):

    """
    Get the dimensions of an object by name.

    Parameters:
    object_name (str): The name of the object.

    Returns:
    list: A list containing the width, height, and depth of the object.
    """
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise ValueError(f"Object '{object_name}' not found.")
    dimensions = obj.dimensions
    return [dimensions.x, dimensions.y, dimensions.z]

def get_modified_mesh(obj, cage=False):
    bm = bmesh.new()
    bm.from_object(
        obj,
        context.evaluated_depsgraph_get(),
        cage=cage,
    )
    bm.verts.ensure_lookup_table()
    return bm

def get_vertex_group_weights(obj):
    """
    Get vertex group weights for each vertex in the mesh.

    Parameters:
    obj (bpy.types.Object): The Blender object.

    Returns:
    dict: A dictionary mapping vertex indices to their vertex group weights.
    """
    vertex_group_weights = {}
    for vertex in obj.data.vertices:
        vertex_group_weights[vertex.index] = {}
        for group in vertex.groups:
            group_name = obj.vertex_groups[group.group].name
            vertex_group_weights[vertex.index][group_name] = group.weight
    return vertex_group_weights

def get_max_coordinates_with_deform(obj, vertex_group_name):
    """
    Get the maximum coordinates (X, Y, Z) of vertices in a vertex group with shape keys applied.

    Parameters:
    obj (bpy.types.Object): The Blender object containing the vertex group.
    vertex_group_name (str): The name of the vertex group.

    Returns:
    list: A list containing the maximum X, Y, and Z coordinates.
    """
    vertex_group = obj.vertex_groups.get(vertex_group_name)
    if vertex_group is None:
        raise ValueError(f"Vertex group '{vertex_group_name}' not found in object '{obj.name}'.")

    bm = get_modified_mesh(obj)
    vertex_group_weights = get_vertex_group_weights(obj)

    max_x = max_y = max_z = None
    for v in bm.verts:
        if vertex_group_name in vertex_group_weights[v.index]:
            co = obj.matrix_world @ v.co  # Use evaluated mesh coordinates
            if max_x is None or co.x > max_x:
                max_x = co.x
            if max_y is None or co.y > max_y:
                max_y = co.y
            if max_z is None or co.z > max_z:
                max_z = co.z

    return [max_x, max_y, max_z]


# Function to compare vertex group distances
def compare_vertex_group_distances(obj, vg1, vg2):
    
    """
    Get the dimensions of an object by name.

    Parameters:
    object: The object that want to compare.
    vg1 : Name of vertex group
    vg2 : Name of vertex group
    Returns:
    list: A list containing the distance of coordination (x,y,z) of the object.
    """
    group1 = obj.vertex_groups.get(vg1)
    group2 = obj.vertex_groups.get(vg2)
    
    if group1 is None:
        raise ValueError(f"Vertex group '{vg1}' not found in object '{obj.name}'.")
    if group2 is None:
        raise ValueError(f"Vertex group '{vg2}' not found in object '{obj.name}'.")

    max_coords_vg1 = get_max_coordinates_with_deform(obj, vg1)
    max_coords_vg2 = get_max_coordinates_with_deform(obj, vg2)


    if None in max_coords_vg1:
        raise ValueError(f"No vertices found in vertex group '{vg1}'.")
    if None in max_coords_vg2:
        raise ValueError(f"No vertices found in vertex group '{vg2}'.")

    distance_x = abs(max_coords_vg1[0] - max_coords_vg2[0])
    distance_y = abs(max_coords_vg1[1] - max_coords_vg2[1])
    distance_z = abs(max_coords_vg1[2] - max_coords_vg2[2])

    return [distance_x, distance_y, distance_z]


# Function to find the nearest vertices to a given point, prioritizing x-distance
def find_nearest_vertices(obj, target_co, num_vertices=5, weight_x=1.0, weight_y=1.0, weight_z=1.0):

    """
    Finds the nearest vertices to a specified target coordinate in a given object using a weighted distance approach.

    Parameters:
    obj (bpy.types.Object): The Blender object containing the mesh from which to find the nearest vertices.
    target_co (mathutils.Vector): The target coordinate to which the nearest vertices are found.
    num_vertices (int, optional): The number of nearest vertices to return. Default is 5.
    weight_x (float, optional): The weight factor for the x-coordinate in distance calculations. Default is 1.0.
    weight_y (float, optional): The weight factor for the y-coordinate in distance calculations. Default is 1.0.
    weight_z (float, optional): The weight factor for the z-coordinate in distance calculations. Default is 1.0.

    Returns:
    List[int]: A list of indices of the nearest vertices to the target coordinate.
    """
    mesh = obj.data
    kd = mathutils.kdtree.KDTree(len(mesh.vertices))
    
    # Scale target coordinates
    scaled_target_co = mathutils.Vector((target_co.x * weight_x, target_co.y * weight_y, target_co.z * weight_z))
    
    # Fill the KDTree with scaled vertex coordinates
    for i, v in enumerate(mesh.vertices):
        scaled_co = mathutils.Vector((v.co.x * weight_x, v.co.y * weight_y, v.co.z * weight_z))
        kd.insert(obj.matrix_world @ scaled_co, i)
    
    kd.balance()
    
    # Find the nearest vertices to the scaled target coordinate
    nearest_vertices = kd.find_n(scaled_target_co, num_vertices)
    
    # # Print debug information
    # print(f"Nearest {num_vertices} vertices to point {target_co}:")
    # for co, index, dist in nearest_vertices:
    #     print(f"Vertex index: {index}, Vertex coordinates: {co}, Distance: {dist}")
    
    return [index for co, index, dist in nearest_vertices]

# Function to scale an object
def scale_object(object_name, new_size, dimension,old_size):

    """
    Scales a specified object in Blender to a new size along a given dimension.

    Parameters:
    object_name (str): The name of the object to be scaled.
    new_size (float): The desired new size for the specified dimension.
    dimension (str): The dimension along which the object will be scaled ('X', 'Y', or 'Z').
    old_size (float): The original size of the object, used for proportional calculations.

    Raises:
    ValueError: If the object is not found or an invalid dimension is provided.

    Explanation:
    - The function retrieves the object by its name and checks if it exists in the Blender data.
    - It calculates the current dimensions of the object using custom helper functions.
    - Depending on the specified dimension, it determines the current size for scaling.
    - The scale factor is calculated based on the desired new size and current size.
    - The object is then scaled by multiplying its scale in the specified dimension by the scale factor.
    - The Blender view layer is updated to reflect the changes.
    """
    
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise ValueError(f"Object '{object_name}' not found.")
    
    current_dimensions = get_object_dimensions(object_name)
    getA = get_max_coordinates_with_deform(obj,"Point_A")
    getA_X = getA[0]
    getF = compare_vertex_group_distances(obj,"Leftmost","Rightmost")
    getF_Y = getF[1]
  

    dimension = dimension.upper()
    if dimension == 'X':
        current_size = getA_X
        scale_index = 0
    elif dimension == 'Y':
        current_size = getF_Y
        scale_index = 1
    elif dimension == 'Z':
        current_size = current_dimensions[2]
        scale_index = 2
    else:
        raise ValueError(f"Invalid dimension '{dimension}'. Choose 'X', 'Y', or 'Z'.")

    scale_factor = (new_size * 10) / current_size  # Adjust scaling factor as needed
    print(scale_factor)
    
    obj.scale[scale_index] = scale_factor
    current_scale = obj.scale.y
    print(f"after apply new scale the current scale is {current_scale}")
    if current_scale == scale_factor:
        print(f"Object '{obj.name}' is already scaled to the expected size in the {dimension} dimension.")

    bpy.context.view_layer.update()

# Function to set shape key value
def set_shape_key_value(obj, shape_key_name, x_blender):
    """
    Sets the value of a shape key for a given object based on the input size in Blender units.

    Parameters:
    obj (bpy.types.Object): The Blender object whose shape key value needs to be set.
    shape_key_name (str): The base name of the shape key to be modified.
    x_blender (float): The size in Blender units for determining the shape key value.

    Explanation:
    - Converts Blender units to real-life centimeters using a predefined ratio.
    - Determines the shape key type (increase or decrease) based on the size input.
    - Limits the shape key value between 0.001 and 1.000 for safety.
    - Sets the shape key's value if it exists in the object's shape key data.
    - Provides informative messages for out-of-range values and non-existent shape keys.
    """
    blender_to_real_ratio = 10.0
    x_real_life_cm = x_blender / blender_to_real_ratio

    if 0.1 <= x_blender <= 10.0:
        shape_key_type = "Inc"
        shape_key_value = x_real_life_cm
    elif x_blender > 10.0:
        shape_key_type = "Inc"
        shape_key_value = 1.0
        print("Exceed 1 cm in real life")
    elif -10.0 <= x_blender <= -0.1:
        shape_key_type = "Dec"
        shape_key_value = abs(x_real_life_cm)
    elif x_blender < -10.0:
        shape_key_type = "Dec"
        shape_key_value = 1.0
        print("Exceed 1 cm in real life")
    else:
        print("Input value is out of the expected range.")
        return

    shape_key_value = max(0.001, min(1.000, shape_key_value))
    full_shape_key_name = f"{shape_key_name}_{shape_key_type}"

    if obj.data.shape_keys:
        shape_key_block = obj.data.shape_keys.key_blocks.get(full_shape_key_name)
        if shape_key_block:
            shape_key_block.value = shape_key_value
            print(f"Set shape key '{full_shape_key_name}' to value: {shape_key_value}")
        else:
            print(f"Shape key '{full_shape_key_name}' not found.")
    else:
        print(f"No shape keys found for object '{obj.name}'.")

def set_shape_key_value_z(obj, shape_key_name, x_blender):
    """
    Sets the value of a shape key for a given object based on the input size in Blender units.

    Parameters:
    obj (bpy.types.Object): The Blender object whose shape key value needs to be set.
    shape_key_name (str): The base name of the shape key to be modified.
    x_blender (float): The size in Blender units for determining the shape key value.

    Explanation:
    - Converts Blender units to real-life centimeters using a predefined ratio.
    - Determines the shape key type (increase or decrease) based on the size input.
    - Limits the shape key value between 0.001 and 1.000 for safety.
    - Sets the shape key's value if it exists in the object's shape key data.
    - Provides informative messages for out-of-range values and non-existent shape keys.
    """
    blender_to_real_ratio = 20.0
    z_real_life_cm = x_blender / blender_to_real_ratio

    if 0.1 <= x_blender <= 20.0:
        shape_key_type = "Inc"
        shape_key_value = z_real_life_cm
    elif x_blender > 10.0:
        shape_key_type = "Inc"
        shape_key_value = 1.0
        print("Exceed 1 cm in real life")
    elif -20.0 <= x_blender <= -0.1:
        shape_key_type = "Dec"
        shape_key_value = abs(z_real_life_cm)
    elif x_blender < -10.0:
        shape_key_type = "Dec"
        shape_key_value = 1.0
        print("Exceed 2 cm in real life")
    else:
        print("Input value is out of the expected range.")
        return

    shape_key_value = max(0.001, min(1.000, shape_key_value))
    full_shape_key_name = f"{shape_key_name}_{shape_key_type}"

    if obj.data.shape_keys:
        shape_key_block = obj.data.shape_keys.key_blocks.get(full_shape_key_name)
        if shape_key_block:
            shape_key_block.value = shape_key_value
            print(f"Set shape key '{full_shape_key_name}' to value: {shape_key_value}")
        else:
            print(f"Shape key '{full_shape_key_name}' not found.")
    else:
        print(f"No shape keys found for object '{obj.name}'.")

def set_shape_key_value_y(obj, shape_key_name_inside, shape_key_name_outside, y_blender):

    """
    Sets the combined values of the inside and outside shape keys based on the specified value in Blender units for the Y axis.

    Args:
    obj: The Blender object with the shape keys.
    shape_key_name_inside: The name of the inside shape key.
    shape_key_name_outside: The name of the outside shape key.
    y_blender: The y-coordinate value in Blender units.
    """
    blender_to_real_ratio = 10.0
    y_real_life_cm = y_blender / blender_to_real_ratio

    if 0.1 <= y_blender <= 20.0:
        shape_key_type = "Inc"
        shape_key_value = y_real_life_cm / 2.0  # Split the value between inside and outside
    elif y_blender > 20.0:
        shape_key_type = "Inc"
        shape_key_value = 1.0  # Max value for each shape key
        print("Exceed 2 cm in real life")
    elif -20.0 <= y_blender <= -0.1:
        shape_key_type = "Dec"
        shape_key_value = abs(y_real_life_cm) / 2.0  # Split the value between inside and outside
    elif y_blender < -20.0:
        shape_key_type = "Dec"
        shape_key_value = 1.0  # Max value for each shape key
        print("Exceed 2 cm in real life")
    else:
        print("Input value is out of the expected range.")
        return

    shape_key_value = max(0.001, min(1.000, shape_key_value))
    full_shape_key_name_inside = f"{shape_key_name_inside}_{shape_key_type}_Inside"
    full_shape_key_name_outside = f"{shape_key_name_outside}_{shape_key_type}_Outside"

    if obj.data.shape_keys:
        shape_key_block_inside = obj.data.shape_keys.key_blocks.get(full_shape_key_name_inside)
        shape_key_block_outside = obj.data.shape_keys.key_blocks.get(full_shape_key_name_outside)
        if shape_key_block_inside and shape_key_block_outside:
            shape_key_block_inside.value = shape_key_value
            shape_key_block_outside.value = shape_key_value
            print(f"Set shape key '{full_shape_key_name_inside}' and '{full_shape_key_name_outside}' to value: {shape_key_value}")
        else:
            if not shape_key_block_inside:
                print(f"Shape key '{full_shape_key_name_inside}' not found.")
            if not shape_key_block_outside:
                print(f"Shape key '{full_shape_key_name_outside}' not found.")
    else:
        print(f"No shape keys found for object '{obj.name}'.")


def reset_shape_keys(obj_name):
    """
    Resets all shape key values to 0 for the specified object.

    Parameters:
    obj_name (str): The name of the object whose shape keys will be reset.
    """
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        raise ValueError(f"Object '{obj_name}' not found.")
    
    if obj.data.shape_keys:
        for shape_key in obj.data.shape_keys.key_blocks:
            shape_key.value = 0.0
        print(f"All shape keys for object '{obj_name}' have been reset to 0.")
    else:
        print(f"No shape keys found for object '{obj_name}'.")

def delete_all_vertex_groups_except(obj_name, exceptions=[]):
    """
    Deletes all vertex groups except for the specified exceptions.

    Parameters:
    obj_name (str): The name of the object.
    exceptions (list): List of vertex group names to be excluded from deletion.
    """
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        raise ValueError(f"Object '{obj_name}' not found.")

    vertex_groups = obj.vertex_groups
    for group in vertex_groups:
        if group.name not in exceptions:
            vertex_groups.remove(group)
    print(f"All vertex groups except {exceptions} have been deleted for object '{obj_name}'.")

def reset_object_scale(obj_name):
    """
    Resets the scale of the object to 0 for all dimensions.

    Parameters:
    obj_name (str): The name of the object.
    """
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        raise ValueError(f"Object '{obj_name}' not found.")
    
    # Reset the scale to 1 for all dimensions
 
    obj.scale[0] = 1.0
    obj.scale[1] = 1.0
    obj.scale[2] = 1.0
    
    current_scale = obj.scale

    current_dimensions = get_object_dimensions(object_name)
    # getA = get_max_coordinates_with_deform(obj,"All_finngers")
    print(current_dimensions)


    print(f"Scale of object '{obj_name}' has been reset to 1 for all dimensions.")
    print(f"Current scale of object '{obj_name}': X = {current_scale[0]}, Y = {current_scale[1]}, Z = {current_scale[2]}")

    current_dimensions = get_object_dimensions(object_name)
    # getA = get_max_coordinates_with_deform(obj,"All_finngers")
    print(current_dimensions)

def delete_shape_keys(obj_name, shape_key_names):
    # Get the object by name
    obj = bpy.data.objects.get(obj_name)
    
    # If the object is not found, do nothing
    if not obj:
        print(f"Object '{obj_name}' not found. No action taken.")
        return
    
    # If the object has no shape keys, do nothing
    if not obj.data.shape_keys:
        print(f"Object '{obj_name}' has no shape keys. No action taken.")
        return
    
    shape_keys = obj.data.shape_keys.key_blocks
    
    # Iterate through the list of shape key names
    for shape_key_name in shape_key_names:
        # If the shape key is not found, skip to the next one
        if shape_key_name not in shape_keys:
            print(f"Shape key '{shape_key_name}' not found in object '{obj_name}'. Skipping...")
            continue
        
        # Find the index of the shape key
        shape_key_index = shape_keys.keys().index(shape_key_name)
        
        # Set the active shape key index to the one you want to delete
        obj.active_shape_key_index = shape_key_index
        
        # Remove the shape key
        bpy.ops.object.shape_key_remove(all=False)
        
        print(f"Shape key '{shape_key_name}' has been deleted from object '{obj_name}'.")
    

def create_shape_key_with_proportion(obj_name, shapekey_name, proportion_size, distance, dimension, vertex_group):
    # Get the object by name
    obj = bpy.data.objects.get(obj_name)
    
    # Check if the object exists
    if obj is None:
        raise ValueError(f"Object '{obj_name}' not found.")
    
    # Ensure the object has the vertex group
    if vertex_group not in obj.vertex_groups:
        raise ValueError(f"Vertex group '{vertex_group}' not found in object '{obj_name}'.")
    
    # Set the object as the active object
    bpy.context.view_layer.objects.active = obj
    
    # Create a new shape key
    obj.shape_key_add(name=shapekey_name)
    obj.active_shape_key_index = len(obj.data.shape_keys.key_blocks) - 1
    
    # Switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Select the vertex group
    bpy.ops.object.vertex_group_set_active(group=vertex_group)
    bpy.ops.object.vertex_group_select()

    # Enable proportional editing and set proportion size
    bpy.context.tool_settings.use_proportional_edit = True
    bpy.context.tool_settings.proportional_edit_falloff = 'SMOOTH'
    bpy.context.tool_settings.proportional_size = proportion_size

    # Check if proportional editing is active
    if not bpy.context.tool_settings.use_proportional_edit:
        raise RuntimeError("Proportional editing failed to activate.")
    
    # Move the selected vertices along the specified dimension
    if dimension.upper() == 'X':
        bpy.ops.transform.translate(value=(distance, 0, 0), orient_type='GLOBAL',use_proportional_edit = True,
                            proportional_edit_falloff='SMOOTH',
                            proportional_size=proportion_size)
    elif dimension.upper() == 'Y':
        bpy.ops.transform.translate(value=(0, distance, 0), orient_type='GLOBAL',use_proportional_edit = True,
                            proportional_edit_falloff='SMOOTH',
                            proportional_size=proportion_size)
    elif dimension.upper() == 'Z':
        bpy.ops.transform.translate(value=(0, 0, distance), orient_type='GLOBAL',use_proportional_edit = True,
                            proportional_edit_falloff='SMOOTH',
                            proportional_size=proportion_size)
    else:
        raise ValueError("Dimension must be 'X', 'Y', or 'Z'")
    
    # Exit edit mode and return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Disable proportional editing to clean up
    bpy.context.tool_settings.use_proportional_edit = False

    print(f"Shape key '{shapekey_name}' created successfully with proportional editing.")

def reset_all(obj_name, vertex_group_exceptions=[],shapekeys=[]):
    """
    Resets the shape keys, deletes all vertex groups except for specified exceptions,
    resets the object scale to 0, and deletes all vertices except those in specified vertex groups.

    Parameters:
    obj_name (str): The name of the object.
    vertex_group_exceptions (list): List of vertex group names to be excluded from deletion.
    """
    delete_shape_keys(obj_name,shapekeys)
    reset_shape_keys(obj_name)
    delete_all_vertex_groups_except(obj_name, vertex_group_exceptions)
    reset_object_scale(obj_name)

# Example usage



# Main morphing function
def morphing_sole(obj_name, old_size_dict, new_size_dict, morphing_keys):

    if bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    print("Start morphing object ")
    obj = bpy.data.objects[obj_name]
    
    try:
        dimension = 'X'
        print("Im try to scale the object")
        scale_object(obj_name, new_size_dict['A'], dimension,old_size_dict)
        print(f"Object '{obj_name}' scaled successfully in the {dimension} dimension.")
    except ValueError as e:
        print(e)
    
    for key in morphing_keys:
        old_size = old_size_dict[key]
        new_size = new_size_dict[key]
        
        vertex_group_name = f"Point_{key}"
        max_coords = get_max_coordinates_with_deform(obj, vertex_group_name)
        max_x = max_coords[0]
        
        if max_x is not None:
            target_x = new_size * 10
            
            print(f"current position of key_{key} {max_x/10} change to the {target_x/10} ")
            shape_key_value = (target_x - max_x)
            set_shape_key_value(obj, key, shape_key_value)


def morphing_Arch(obj_name, old_size_dict, new_size_dict, morphing_keys):

    if bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    print("Start morphing object ")
    obj = bpy.data.objects[obj_name]
    
    
    for key in morphing_keys:
        old_size = old_size_dict[key]
        new_size = new_size_dict[key]
        
        vertex_group_name = f"Point_{key}"
        max_coords = get_max_coordinates_with_deform(obj, vertex_group_name)
        max_x = max_coords[2]
        
        if max_x is not None:
            target_x = new_size * 10
            print(f"current position of key_{key} {max_x/10} change to the {target_x/10} ")
            shape_key_value = (target_x - max_x)
            set_shape_key_value_z(obj, key, shape_key_value)



# Function to create a vertex group for a specific line
def create_vertex_group(obj, name, target_co, num_vertices=5, weight_x=1.0, weight_y=1.0, weight_z=1.0):
    # Ensure Blender is in Object Mode
    if bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Find the nearest vertices to the initial point
    nearest_vertices_indices = find_nearest_vertices(obj, target_co, num_vertices, weight_x, weight_y, weight_z)
    
    # Create a new vertex group and add these vertices to the group
    vg = obj.vertex_groups.new(name=name)
    vg.add(nearest_vertices_indices, 1.0, 'ADD')    

def get_vertex_group_coordinates(obj, group_name):
    # Ensure Blender is in Object Mode
    if bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Get the vertex group
    vg = obj.vertex_groups.get(group_name)
    
    if vg is None:
        print(f"Vertex group {group_name} not found")
        return
    
    # Get the mesh data
    mesh = obj.data
    
    # Find the vertices in the vertex group
    vertices = [v for v in mesh.vertices if vg.index in [vg.group for vg in v.groups]]
    
    # Print the coordinates of the vertices in the vertex group
    print(f"Coordinates of vertices in vertex group {group_name}:")
    for v in vertices:
        print(obj.matrix_world @ v.co)



# Example usage:
# Main function to execute the script
def main_sole(obj_name,model_dict):
    # Set these variables
    object_name = obj_name  # Replace with your object's name
    value_dict = model_dict


    # Weights for prioritizing x-distance
    weight_x = 40.0
    weight_y = 3.0
    weight_z = 3.0
    # Get the object
    obj = bpy.data.objects[object_name]

    num_vertices = 3  # Number of nearest vertices to select


    # Mark vertex groupfor Point A 
    vg_point_A = 'A' 

    # Check if the key is in the dictionary and create the vertex group
    if vg_point_A in value_dict:
        value = value_dict[vg_point_A]
        target_x = value  # Real-life length in CM
        new_x = get_max_coordinates_with_deform(obj,"All_fingers")
        target_co = mathutils.Vector((new_x[0], 0, new_x[2]/2))  # Convert to Blender units (1 CM = 10 M)
        group_name = f"Point_{vg_point_A}"  # Name of the vertex group
        create_vertex_group(obj, group_name, target_co, num_vertices, weight_x, weight_y, weight_z)
    else:
        print(f"Key '{vg_point_A}' not found in value_dict.")

        # Mark leftmost and rightmost vertices along Y axis


    vg_point_soleFoot = ['B', 'I']  # Keys of the values to create vertex groups for point B D E I
    # Create vertex groups for specified values
    for key in vg_point_soleFoot:

        value = value_dict[key]
        target_x = value  # Real-life length in CM
        target_co = mathutils.Vector((target_x * 10, 0, 0))  # Convert to Blender units (1 CM = 10 M)
        group_name = f"Point_{key}"  # Name of the vertex group
        create_vertex_group(obj, group_name, target_co, num_vertices)
    
    vg_point_D = ['D']  # Keys of the values to create vertex groups for point B D E I
    # Create vertex groups for specified values
    for key in vg_point_D:

        value = value_dict[key]
        target_x = value  # Real-life length in CM
        target_co = mathutils.Vector((target_x * 10, 35, 10))  # Convert to Blender units (1 CM = 10 M)
        group_name = f"Point_{key}"  # Name of the vertex group
        create_vertex_group(obj, group_name, target_co, num_vertices)

    vg_point_E = ['E']  # Keys of the values to create vertex groups for point B D E I
    # Create vertex groups for specified values
    for key in vg_point_E:

        value = value_dict[key]
        target_x = value  # Real-life length in CM
        target_co = mathutils.Vector((target_x * 10, -55, 10))  # Convert to Blender units (1 CM = 10 M)
        group_name = f"Point_{key}"  # Name of the vertex group
        create_vertex_group(obj, group_name, target_co, num_vertices)
    
    vg_point_D = ['D']  
    # Create vertex groups for specified values
    for key in vg_point_D:

        num_vertices = 1

        targeto = get_max_coordinates_with_deform(obj,f"Point_{key}")
        target_x = targeto[0]  # Real-life length in CM
        target_co = mathutils.Vector((target_x , 100, 15))  # Convert to Blender units (1 CM = 10 M)
        group_name = "Rightmost" # Name of the vertex group
        create_vertex_group(obj, group_name, target_co, num_vertices)

    vg_point_E = ['E']  
    # Create vertex groups for specified values
    for key in vg_point_E:

        num_vertices = 1
        targeto = get_max_coordinates_with_deform(obj,f"Point_{key}")
        target_x = targeto[0]  # Real-life length in CM
        target_co = mathutils.Vector((target_x , -100, 15))  # Convert to Blender units (1 CM = 10 M)
        group_name = f"Leftmost"  # Name of the vertex group
        create_vertex_group(obj, group_name, target_co, num_vertices)

    H = get_max_coordinates_with_deform(obj,"Point_I")
    H_x = H[0]
    vg_point_H = 'H'      


    value = value_dict[vg_point_H]
    # print(value)
    target_Z = value*10  # Real-life length in CM
    print(target_Z)
    target_co = mathutils.Vector((H_x, 45, target_Z))  # Convert to Blender units (1 CM = 10 M)
    group_name = f"Point_H"  # Name of the vertex group
    create_vertex_group(obj, group_name, target_co, weight_x = 40, weight_y = 20.0, weight_z = 100.0)
    print(get_max_coordinates_with_deform(obj,"Point_H"))

        # Create vertex groups for specified values
    num_vertices = 45
    value = value_dict[vg_point_H]
    # print(value)
    target_Z = value*10  # Real-life length in CM
    # print(H,target_Z)
    target_co = mathutils.Vector((H_x, 45, target_Z))  # Convert to Blender units (1 CM = 10 M)
    group_name = f"H"  # Name of the vertex group
    create_vertex_group(obj, group_name, target_co, num_vertices, weight_x = 100, weight_y = 20.0, weight_z = 100.0)
    create_shape_key_with_proportion(obj_name,"H_Inc",45,20,"Z","H")
    create_shape_key_with_proportion(obj_name,"H_Dec",45,-20,"Z","H")


def main_width(obj_name,model_dict):

    object_name = obj_name  # Replace with your object's name
    value_dict = model_dict  # Dictionary of values for A to I in cm

    # Weights for prioritizing x-distance
    weight_x = 40.0
    weight_y = 3.0
    weight_z = 5.0
    # Get the object
    obj = bpy.data.objects[object_name]

    num_vertices = 5  # Number of nearest vertices to select

    vg_point_D = ['D']  
    # Create vertex groups for specified values
    for key in vg_point_D:
        num_vertices = 30
        targeto = get_max_coordinates_with_deform(obj,f"Point_{key}")
        target_x = targeto[0]  # Real-life length in CM
        target_co = mathutils.Vector((target_x , 70, 15))  # Convert to Blender units (1 CM = 10 M)
        group_name = f"F_Inside"  # Name of the vertex group
        create_vertex_group(obj, group_name, target_co, num_vertices)

    vg_point_E = ['E']  
    # Create vertex groups for specified values
    for key in vg_point_E:
        num_vertices = 30
        targeto = get_max_coordinates_with_deform(obj,f"Point_{key}")
        target_x = targeto[0]  # Real-life length in CM
        target_co = mathutils.Vector((target_x , -70, 15))  # Convert to Blender units (1 CM = 10 M)
        group_name = f"F_Outside"  # Name of the vertex group
        create_vertex_group(obj, group_name, target_co, num_vertices)


    vg_point_C_Inside = ['I']  
    # Create vertex groups for specified values
    for key in vg_point_C_Inside:
        num_vertices = 30
        targeto = get_max_coordinates_with_deform(obj,f"Point_{key}")
        tar_x = targeto[0]
        target_co = mathutils.Vector((tar_x, 70, 15))  # Convert to Blender units (1 CM = 10 M)
        group_name = f"C_Inside"  # Name of the vertex group
        create_vertex_group(obj, group_name, target_co, num_vertices)

    vg_point_C_Outside = ['I']  # Keys of the values to create vertex groups for point B D E I
    # Create vertex groups for specified values
    for key in vg_point_C_Outside:   
        num_vertices = 30
        # value = value_dict[key]
        targeto = get_max_coordinates_with_deform(obj,f"Point_{key}")
        tar_x = targeto[0]
        # target_x = value/2  # Real-life length in CM
        target_co = mathutils.Vector((tar_x , -70, 15))  # Convert to Blender units (1 CM = 10 M)
        group_name = f"C_Outside"  # Name of the vertex group
        create_vertex_group(obj, group_name, target_co, num_vertices)

    vg_point_G_Outside = 4.7  # Real-life length in CM
    # Create vertex group for vg_point_G_Left

    num_vertices = 30
    target_co = mathutils.Vector((vg_point_G_Outside * 10, -27, 15))  # Convert to Blender units (1 CM = 10 M)
    group_name = f"G_Outside"  # Name of the vertex group
    create_vertex_group(obj, group_name, target_co, num_vertices)

    vg_point_G_Inside = 4.7  # Real-life length in CM
    # Create vertex group for vg_point_G_Right
    num_vertices = 30
    target_co = mathutils.Vector((vg_point_G_Inside * 10, 27, 15))  # Convert to Blender units (1 CM = 10 M)
    group_name = f"G_Inside"  # Name of the vertex group
    create_vertex_group(obj, group_name, target_co, num_vertices)

    # start = get_max_coordinates_with_deform(obj,'C_Inside')
    # end = get_max_coordinates_with_deform(obj,'G_Inside')
    # H = (start[0]+ end[0])/2



# Main morphing function
def morphing_width(obj_name, old_size_dict, new_size_dict, morphing_keys):

    if bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    print("Start morphing object ")
    obj = bpy.data.objects[obj_name]


    try:
        dimension = 'Y'
        scale_object(obj_name,new_size_dict['F'], dimension,old_size_dict)
        print(f"Object '{obj_name}' scaled successfully in the {dimension} dimension.")
    except ValueError as e:
        print(e)
    
    for key in morphing_keys:
       
        old_size = old_size_dict[key]
        new_size = new_size_dict[key]
        vg_1 = f"{key}_Inside"
        vg_2 = f"{key}_Outside"       
        dis = compare_vertex_group_distances(obj, vg_1, vg_2)
        dis_y = dis[1]
        
        if dis_y is not None:
            target_x = new_size * 10
            print(f"current position of key_{key} {dis_y/10} change to the {target_x/10} ")
            shape_key_value = (target_x - dis_y)
            set_shape_key_value_y(obj, key, key,shape_key_value)

def main():

    if bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    object_name = 'AI-001-SUTHAWAN_BUCHATHIP_Right'
    obj = bpy.data.objects.get(object_name)
    except_groups = ["All_fingers", "B","D","E","I"]  # Replace with the names of vertex groups you want to keep
    deleted_shapekeys = ["H_Inc","H_Dec"]
    # reset_shape_keys(object_name)
    # reset_object_scale(object_name)
    # delete_all_vertex_groups_except(object_name,except_groups)

    reset_all(object_name, except_groups,deleted_shapekeys)
    # print("reset ALL to the default")


    old_size_dict = {
        'A': 23.58,
        'B': 19.795,
        'C': 8.11,
        'D': 17.492,
        'E': 15.211,
        'F': 9.365,
        'G': 5.36,
        'H': 3.96,
        'I': 9.898
    }

    new_size_dict = {
        'A': 25.827,
        'B': 21.605,
        'C': 8.89,
        'D': 18.979,
        'E': 16.384,
        'F': 10.422,
        'G': 6.401,
        'H': 5.15,
        'I': 10.803
    }


    morphing_keys = ['A','B', 'D', 'E', 'I']
    morphing_width_keys = ['C','F','G']
    morphing_Arch_keys = ['H']
    

    # reset_all(object_name, except_groups)

    # Function call
    main_sole(object_name,old_size_dict)
    morphing_sole(object_name, old_size_dict, new_size_dict, morphing_keys)

    main_width(object_name,old_size_dict)
    morphing_width(object_name, old_size_dict, new_size_dict, morphing_width_keys)

    morphing_Arch(object_name, old_size_dict, new_size_dict, morphing_Arch_keys)

    result_check = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    z_dimension = ['H']
    x_dimension = ['A', 'B', 'D', 'E', 'I']
    y_dimension = ['C', 'F', 'G']
    all_results = []
    for ck in result_check:
        if obj:
            if ck in x_dimension:
                # Get the maximum x-coordinate of the specified vertex group
                max_coords = get_max_coordinates_with_deform(obj, f'Point_{ck}')
                max_x = max_coords[0]/10
                # Print the maximum x-coordinate
                all_results.append(max_x)
                print(f"The maximum x-coordinate of the vertex group 'Point_{ck}' is: {max_x }")

            elif ck in y_dimension:
                 # Compare vertex group distances for inside and outside vertex groups
                vg_1 = f"{ck}_Inside"
                vg_2 = f"{ck}_Outside"
                max_coords = compare_vertex_group_distances(obj, vg_1, vg_2)
                max_y = max_coords[1]/10
                all_results.append(max_y)
                # Print the maximum y-coordinate
                print(f"The maximum y-coordinate of the vertex group 'Point_{ck}' is: {max_y}")

            elif ck in z_dimension:
                # Get the maximum z-coordinate of the specified vertex group
                max_coords = get_max_coordinates_with_deform(obj, f'Point_{ck}')
                max_z = max_coords[2]/10
                all_results.append(max_z)
                # Print the maximum z-coordinate
                print(f"The maximum z-coordinate of the vertex group 'Point_{ck}' is: {max_z}")
        else:
            print(f"Object '{object_name}' not found")
    return all_results



object_name = 'AI-001-SUTHAWAN_BUCHATHIP_Right'
obj = bpy.data.objects.get(object_name)
except_groups = ["All_fingers", "B","D","E","I"]  # Replace with the names of vertex groups you want to keep
# Run the script

# reset_shape_keys(object_name)

# reset_all(object_name,except_groups)
# x = get_max_coordinates_with_deform(obj,"All_fingers")
# print(x)
# y = compare_vertex_group_distances(obj,"F_Inside","F_Outside")
# print(y[1])


main()




# test_result = []
# for i in range(2):
#     ans = main()    
#     test_result.append(ans)

# for t in test_result:
#     print(t)



# reset_shape_keys('AI-001-SUTHAWAN_BUCHATHIP_Left')

