# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 00:22:21 2020

@author: Dawid

TODO:
    szew na sferze
        podzielic sfere na dwa obiekty i osobno teksturowac?
        new fragment shader somehow works and fix it
    
    zlepic 360 z kamer
    
    zeedytowac jako video
    
    restrict camera movement
    add mouse?
    add predifined camera
    
    Running vispy programs in ipython (spyder console) leaves canvas after closing, running multiple times results in memory leak?

"""

import numpy
import cv2
from vispy import app, gloo, io
from vispy.util.transforms import translate, perspective, rotate, scale
from vispy.geometry import create_sphere, create_plane
from vispy.gloo import VertexBuffer, IndexBuffer

vertex = """
#version 120

// Uniforms
// ------------------------------------
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform sampler2D texture;

// Attributes
// ------------------------------------
attribute vec3 position;
attribute vec2 texcoord;
attribute vec3 normal;
attribute vec4 color;

// Varyings
// ------------------------------------
varying vec3 v_position;
varying vec2 v_texture_coords;
varying vec4 v_color;

void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
    v_position = position;
    v_texture_coords = texcoord;
    v_color = color;
}
"""

fragment_sphere = """
#version 120

// Constants
// ------------------------------------
const float M_PI = 3.14159265358979323846;
const float INFINITY = 1000000000.;

// Varyings
// ------------------------------------
varying vec3 v_position;
varying vec2 v_texture_coords;
//varying vec4 v_color;

// Uniforms
// ------------------------------------
uniform sampler2D texture;

// Functions
// ------------------------------------

// Main
// ------------------------------------
void main() {
    vec2 tc = v_texture_coords;
    tc.x = (M_PI + atan(v_position.y, v_position.x)) / (2 * M_PI); // calculate angle and map it to 0..1
    gl_FragColor = texture2D(texture, tc);
}
"""

fragment_rectangle = """
#version 120

// Constants
// ------------------------------------
const float M_PI = 3.14159265358979323846;
const float INFINITY = 1000000000.;

// Varyings
// ------------------------------------
varying vec3 v_position;
varying vec2 v_texture_coords;
//varying vec4 v_color;

// Uniforms
// ------------------------------------
uniform sampler2D texture;

// Functions
// ------------------------------------

// Main
// ------------------------------------
void main() {
    gl_FragColor = texture2D(texture, v_texture_coords);
}
"""

fragment_car = """
#version 120

// Constants
// ------------------------------------
const float M_PI = 3.14159265358979323846;
const float INFINITY = 1000000000.;

// Varyings
// ------------------------------------
varying vec3 v_position;
varying vec2 v_texture_coords;
varying vec4 v_color;

// Uniforms
// ------------------------------------
uniform sampler2D texture;

// Functions
// ------------------------------------

// Main
// ------------------------------------
void main() {
    gl_FragColor = v_color;
}
"""

def create_camera_matrix(azimuthal_angle, polar_angle, translation_y = 0, translation_z = 0):
    # First translate matix, then rotate it around center
    rotate_around_center = numpy.matmul(rotate(azimuthal_angle, (0, 0, 1)),
                                       rotate(polar_angle, (1, 0, 0)))
    translate_matrix = translate((0, translation_y, translation_z))
    return numpy.matmul(rotate_around_center,
                        translate_matrix)
        
def look_at():
    pass

def checkerboard(grid_num=8, grid_size=32):
    row_even = grid_num // 2 * [0, 1]
    row_odd = grid_num // 2 * [1, 0]
    Z = numpy.row_stack(grid_num // 2 * (row_even, row_odd)).astype(numpy.uint8)
    return 255 * Z.repeat(grid_size, axis=0).repeat(grid_size, axis=1)

#https://www.programcreek.com/python/?code=kirumang%2FPix2Pose%2FPix2Pose-master%2Frendering%2Frenderer.py#
# przepisać pod swoje potrzeby :)
# tam jest mowa o face-wise i vertex-wise cokolwiek to znaczy
def load_texture(filename):
    # print('Loading {}'.format(filename))
    # image = cv2.flip(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 0)  # Must be flipped because of OpenGL
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    return image
    # return gloo.Texture2D(image, format='rgb')

def random_uv(number_of_vertices):
    return numpy.random.rand(number_of_vertices,2).astype(numpy.float32)

def color(number_of_vertices, color = None):
    if color is None:
        C = numpy.random.rand(size=(number_of_vertices,4),dtype=numpy.float32)
        C[:,3] = 1.0
        return C
    else:
        return numpy.full(shape=(number_of_vertices,4), fill_value = color, dtype=numpy.float32)

def sphere_uv(vertices, radius=None):
    if radius == None:
        radius = numpy.average(numpy.linalg.norm(vertices, axis=1))
    normalized_vertices = vertices / radius
    u = numpy.arctan2(normalized_vertices[:,1],normalized_vertices[:,0]) / (numpy.pi * 2) + 0.5
    # u = numpy.arctan2(normalized_vertices[:,0],normalized_vertices[:,1]) / (numpy.pi * 2) + 0.5
    v = normalized_vertices[:,2] * 0.5 + 0.5
    # flip last uv
    # u[-1] = 1 - u[-1]
    # v[-1] = 1 - v[-1]
    
    return numpy.transpose(numpy.vstack((u,v)))

class Canvas(app.Canvas):
    def __init__(self):
        def toggle_fs():
            self.fullscreen = not self.fullscreen
        keys = dict(escape='close', F11=toggle_fs, q='close', Q='close')
        app.Canvas.__init__(self, title='Video-Viewer', position=(300, 100),
                            size=(800, 600), keys=keys)

        # Create sphere
        # sphere = create_sphere(rows=10, cols=10, radius=10, offset=True, method='latitude')
        sphere = create_sphere(radius=10, subdivisions=2, method='ico')
        # sphere = create_sphere(radius=10, rows=6, cols=6, method='cube')
        V = sphere.get_vertices().astype(numpy.float32) #converting because probably wont be updated soon
        # T = random_uv(len(V))
        T = sphere_uv(V)
        N = sphere.get_vertex_normals()
        C = sphere.get_vertex_colors() #is NoneType object?
        I = sphere.get_faces().astype(numpy.uint32) #converting because probably wont be updated soon

        # vtype = [('position', numpy.float32, 3),
        #      ('texcoord', numpy.float32, 2),
        #      ('normal', numpy.float32, 3),
        #      ('color',    numpy.float32, 4)]
        
        # vertices = numpy.zeros(len(V), vtype)
        
        vertices = numpy.zeros(len(V),
                    [('position', numpy.float32, 3),
                      ('texcoord', numpy.float32, 2),
                      ('normal', numpy.float32, 3),
                      ('color', numpy.float32, 4)])
    
        vertices['position'] = V
        vertices['texcoord'] = T
        vertices['normal'] = N
        vertices['color'] = C
        
        vertex_buffer_sphere = VertexBuffer(vertices)
        self.indices_sphere = IndexBuffer(I)
        
        # Create rectangle
        vertices_rectangle, I_rectangle, O_rectangle = create_plane(width=20, height=20)

        vertex_buffer_rectangle = VertexBuffer(vertices_rectangle)
        self.indices_rectangle = IndexBuffer(I_rectangle)
        
        camera_height = 1.05
        car_model_scale = 0.03
        
        # Create car obj
        # car_vertices, car_faces, N_car, car_texcoords = io.read_mesh("12353_Automobile_V1_L2.obj")
        car_vertices, car_faces, N_car, car_texcoords = io.read_mesh("CarModel.obj")
        V_car = car_vertices.astype(numpy.float32)
        T_car = random_uv(len(V_car))
        # T = sphere_uv(V) # tekstur nie dawać, jakis jeden kolor wystarczy
        C_car = color(len(V_car), color=(0.124,0.124,0.124,1))
        I_car = car_faces.astype(numpy.uint32)
        
        vertices_car = numpy.zeros(len(V_car),
                    [('position', numpy.float32, 3),
                      ('texcoord', numpy.float32, 2),
                      ('normal', numpy.float32, 3),
                      ('color', numpy.float32, 4)])
    
        vertices_car['position'] = V_car
        vertices_car['texcoord'] = T_car
        vertices_car['normal'] = N_car
        vertices_car['color'] = C_car
        
        vertex_buffer_car = VertexBuffer(vertices_car)
        self.indices_car = IndexBuffer(I_car)
        
        
        # Build program
        self.program_sphere = gloo.Program(vertex, fragment_sphere)
        self.program_sphere.bind(vertex_buffer_sphere)
        
        self.program_rectangle = gloo.Program(vertex, fragment_rectangle)
        self.program_rectangle.bind(vertex_buffer_rectangle)
        #gloo.gl.glUseProgram(self.program_sphere)
        
        self.program_car =  gloo.Program(vertex, fragment_car)
        self.program_car.bind(vertex_buffer_car)
        
        # self.program_sphere['texture'] = checkerboard()
        # self.program_sphere['texture'] = load_texture('1_earth_8k.jpg')
        self.texture_sphere = gloo.Texture2D(load_texture('equirectangular_image_square.jpg'), format='rgb')
        # self.texture_sphere = gloo.Texture2D(load_texture('equirectangular_image.jpg'), format='rgb')
        self.texture_rectangle = gloo.Texture2D(load_texture('top_view_image.jpg'), format='rgb')
        # self.texture_rectangle = gloo.Texture2D(load_texture('top_view_image-5.jpg'), format='rgb')
        # self.texture_sphere = gloo.Texture2D(load_texture('equirectangular_image_square-5.jpg'), format='rgb')
        
        self.program_sphere['texture'] = self.texture_sphere
        self.program_rectangle['texture'] = self.texture_rectangle
        
        # Camera and perspective parameters
        self.distance = 10
        self.height = 2
        self.polar_angle = 90
        self.azimuthal_angle = 0
        self.view = create_camera_matrix(self.azimuthal_angle, self.polar_angle, -self.height, -self.distance) # translate((0, 0, -self.distance))
        self.model = numpy.eye(4, dtype=numpy.float32)
        self.projection = numpy.eye(4, dtype=numpy.float32)
        
        
        self.program_sphere['u_projection'] = self.projection
        self.program_sphere['u_model'] = self.model
        self.program_sphere['u_view'] = self.view
        self.program_rectangle['u_projection'] = self.projection
        # self.program_rectangle['u_model'] = numpy.matmul(rotate(90, (0, 0, 1)),
        #                                                  translate((0, 0, 1.1)))
        self.program_rectangle['u_model'] = numpy.matmul(rotate(180, (0, 0, 1)),
                                                          translate((0, 0, camera_height)))
        self.program_rectangle['u_view'] = self.view
        self.program_car['u_projection'] = self.projection
        self.program_car['u_model'] =  numpy.matmul(numpy.matmul(numpy.matmul(
                                                    scale((car_model_scale,car_model_scale,car_model_scale)),
                                                    rotate(90, (0, 1, 0))), # obrót horyzontalny
                                                    rotate(270, (1, 0, 0))), # obrót góra dół
                                                    translate((0, 0, camera_height)))
        self.program_car['u_view'] = self.view
        
        self.apply_zoom()

        gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True, cull_face=True)
        # gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True,
        #                polygon_offset_fill=True, polygon_offset=(1, 1)) # additional parameters found in https://github.com/vispy/vispy/blob/master/vispy/visuals/sphere.py
        
        self.context.glir.command('FUNC', 'glCullFace', 'front') # invisible triangles will be not drawn, but it does not improve FPS significantly
            #Culling mode. Can be "front", "back", or "front_and_back".
        
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        # self._timer = app.Timer(interval=0, connect=self.on_timer, start=True)
        
        self.draw_timer = 0.0
        self.measure_fps(window=1)

        # self.show()

    def on_timer(self, event):
        # t = event.elapsed
        self.draw_timer += event.dt
        if self.draw_timer > 0.04: # uptade 24 FPS
            self.draw_timer -= 0.04
            self.texture_sphere.set_data(load_texture('equirectangular_image_square.jpg'))
            self.texture_rectangle.set_data(load_texture('top_view_image.jpg'))
            self.program_sphere['texture'] = self.texture_sphere
            self.program_rectangle['texture'] = self.texture_rectangle
            self.update()

    def on_resize(self, event):
        self.apply_zoom()

    def on_mouse_wheel(self, event):
        self.distance -= event.delta[1] / 3
        self.distance = max(6.66, self.distance)
        self.distance = min(12.67, self.distance)
        self.view = create_camera_matrix(self.azimuthal_angle, self.polar_angle, -self.height, -self.distance)
        self.program_sphere['u_view'] = self.view
        self.program_rectangle['u_view'] = self.view
        self.program_car['u_view'] = self.view
        self.update()
        
    def apply_zoom(self):
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(60.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)
        self.program_sphere['u_projection'] = self.projection
        self.program_rectangle['u_projection'] = self.projection
        self.program_car['u_projection'] = self.projection
        self.update()
        
        
    def on_key_press(self, event):
        # modifiers = [key.name for key in event.modifiers]
        # print('Key pressed - text: %r, key: %s, modifiers: %r, type: %s' % (
        #     event.text, event.key.name, modifiers, type(event.key.name)))
        
        # Close program / also Esc as default for interactive
        # if(event.key.name == "Q" or event.key.name == "q"):
        #     self.close()
        
        # Rotate left
        if(event.key.name == "Left" or event.key.name == "A"):
            self.azimuthal_angle += 3

        # Rotate right
        if(event.key.name == "Right" or event.key.name == "D"):
            self.azimuthal_angle -= 3
        
        # Rotate up
        if(event.key.name == "Up" or event.key.name == "W"):
            self.polar_angle -= 3
            
            
        # Rotate down
        if(event.key.name == "Down" or event.key.name == "S"):
            self.polar_angle += 3
        
        # Move up
        if(event.key.name == "R"):
            self.height += 0.3
            
        # Move down
        if(event.key.name == "F"):
            self.height -= 0.3
            
        self.height = max(0, self.height)
        self.height = min(5, self.height)
        self.polar_angle = max(76, self.polar_angle)
        self.polar_angle = min(190, self.polar_angle)
        
        self.view = create_camera_matrix(self.azimuthal_angle, self.polar_angle, -self.height, -self.distance)
        self.program_sphere['u_view'] = self.view
        self.program_rectangle['u_view'] = self.view
        self.program_car['u_view'] = self.view
        self.update()

    def on_draw(self, event):
        # gloo.clear(color=True, depth=True) # does it even work?
        self.context.glir.command('FUNC', 'glCullFace', 'front')
        self.program_sphere.draw('triangles', self.indices_sphere)
        # self.context.glir.command('FUNC', 'glCullFace', 'front_and_back')
        self.program_rectangle.draw('triangles', self.indices_rectangle)
        # self.context.glir.command('FUNC', 'glCullFace', 'front')
        self.program_car.draw('triangles', self.indices_car)
    
    
    # def _remove_programs(self):
    #     self._keep_it_alive = self.program_rectangle
    #     self._keep_it_alive = self.program_sphere
    #     self.program_rectangle = None
    #     self.program_sphere = None
    
    # def on_close(self, event):
        # self.measure_fps(callback=False)
        # self._timer.stop()
        # self._remove_programs()

if __name__ == '__main__':
    canvas = Canvas()
    # canvas.measure_fps()
    canvas.show()
    app.run()
    # canvas.close()
    app.quit()
    