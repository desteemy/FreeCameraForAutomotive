# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 00:22:21 2020

@author: Dawid

TODO:
    szew na sferze
        obejrzeć film
    

"""

import numpy
import cv2
from vispy import app, gloo
from vispy.util.transforms import translate, perspective, rotate
from vispy.geometry import create_sphere
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
attribute vec3 a_position;
attribute vec2 a_texcoord;
attribute vec3 a_normal;
attribute vec4 a_color;

// Varyings
// ------------------------------------
varying vec3 v_position;
varying vec2 v_texture_coords;

void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    v_position = a_position;
    v_texture_coords = a_texcoord;
}
"""

fragment = """
#version 120

// Constants
// ------------------------------------
const float M_PI = 3.14159265358979323846;
const float INFINITY = 1000000000.;

// Varyings
// ------------------------------------
varying vec3 v_position;
varying vec2 v_texture_coords;

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

def create_camera_matrix(azimuthal_angle, polar_angle, translation_z = 0):
    # First translate matix, then rotate it around center
    rotate_around_center = numpy.matmul(rotate(azimuthal_angle, (0, 0, 1)),
                                       rotate(polar_angle, (1, 0, 0)))
    translate_matrix = translate((0, 0, translation_z))
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
    print('Loading {}'.format(filename))
    # image = cv2.flip(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 0)  # Must be flipped because of OpenGL
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    # return image
    return gloo.Texture2D(image, format='rgb')

def random_uv(number_of_vertices):
    return numpy.random.rand(number_of_vertices,2).astype(numpy.float32)

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
        app.Canvas.__init__(self, title='Sphere', position=(300, 100),
                            size=(800, 600), keys='interactive')

        # Create sphere
        # sphere = create_sphere(rows=10, cols=10, radius=10, offset=True, method='latitude')
        sphere = create_sphere(radius=10, subdivisions=5, method='ico') # does not work
        # sphere = create_sphere(rows=10, cols=10, radius=10, method='cube')
        V = sphere.get_vertices()
        #T = random_uv(len(V))
        T = sphere_uv(V)
        N = sphere.get_vertex_normals()
        C = sphere.get_vertex_colors() #is NoneType object?
        I = sphere.get_faces()#.astype(numpy.uint32)

        # vtype = [('a_position', numpy.float32, 3),
        #      ('a_texcoord', numpy.float32, 2),
        #      ('a_normal', numpy.float32, 3),
        #      ('a_color',    numpy.float32, 4)]
        
        # vertices = numpy.zeros(len(V), vtype)
        
        vertices = numpy.zeros(len(V),
                    [('a_position', numpy.float32, 3),
                      ('a_texcoord', numpy.float32, 2),
                      ('a_normal', numpy.float32, 3),
                      ('a_color', numpy.float32, 4)])
    
        vertices['a_position'] = V
        vertices['a_texcoord'] = T
        vertices['a_normal'] = N
        vertices['a_color'] = C
        
        
        vertex_buffer = VertexBuffer(vertices)
        self.indices = IndexBuffer(I)
        
        # Build program
        self.program = gloo.Program(vertex, fragment)
        self.program.bind(vertex_buffer)
        
        # self.program['texture'] = checkerboard()
        self.program['texture'] = load_texture('1_earth_8k.jpg')
        
        # Camera and perspective parameters
        self.translate = 5
        self.polar_angle = 90
        self.azimuthal_angle = 0
        self.view = create_camera_matrix(self.azimuthal_angle, self.polar_angle, -self.translate) # translate((0, 0, -self.translate))
        self.model = numpy.eye(4, dtype=numpy.float32)
        self.projection = numpy.eye(4, dtype=numpy.float32)
        
        
        self.program['u_projection'] = self.projection
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        
        self.apply_zoom()

        gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True)
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        # self._timer = app.Timer('interval=0', connect=self.on_timer, start=True)
        
        # self.measure_fps(window=1)

        self.show()

    def on_timer(self, event):
        #t = event.elapsed
        self.update()

    def on_resize(self, event):
        self.apply_zoom()

    def on_mouse_wheel(self, event):
        self.translate -= event.delta[1] / 3
        self.translate = max(2, self.translate)
        self.view = create_camera_matrix(self.azimuthal_angle, self.polar_angle, -self.translate)
        self.program['u_view'] = self.view
        self.update()
        
    def apply_zoom(self):
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)
        self.program['u_projection'] = self.projection
        
        
    def on_key_press(self, event):
        # modifiers = [key.name for key in event.modifiers]
        # print('Key pressed - text: %r, key: %s, modifiers: %r, type: %s' % (
        #     event.text, event.key.name, modifiers, type(event.key.name)))
        
        # Close program / also Esc as default for interactive
        if(event.key.name == "Q" or event.key.name == "q"):
            self.close()
        
        # Rotate left
        if(event.key.name == "Left"):
            self.azimuthal_angle += 3
            self.view = create_camera_matrix(self.azimuthal_angle, self.polar_angle, -self.translate)
            self.program['u_view'] = self.view
            self.update()
            
        # Rotate right
        if(event.key.name == "Right"):
            self.azimuthal_angle -= 3
            self.view = create_camera_matrix(self.azimuthal_angle, self.polar_angle, -self.translate)
            self.program['u_view'] = self.view
            self.update()
        
        # Rotate up
        if(event.key.name == "Up"):
            self.polar_angle += 3
            self.view = create_camera_matrix(self.azimuthal_angle, self.polar_angle, -self.translate)
            self.program['u_view'] = self.view
            self.update()
            
        # Rotate up
        if(event.key.name == "Down"):
            self.polar_angle -= 3
            self.view = create_camera_matrix(self.azimuthal_angle, self.polar_angle, -self.translate)
            self.program['u_view'] = self.view
            self.update()
        

    def on_draw(self, event):
        # gloo.clear(color=True, depth=True) # does it even work?
        self.program.draw('triangles', self.indices)
    
    # def on_close(self, event):
    #     self.measure_fps(callback=False)
    #     self._timer.stop()

if __name__ == '__main__':
    canvas = Canvas()
    app.run()