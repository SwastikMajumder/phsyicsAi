import math
import itertools
import mathai
import parser
from base import *


command =\
"""move 200
turn -90 -pi/2
move 100
turn 90 pi/2
box m a
turn -90 -pi/2
move 100
turn -90 -pi/2
move 300"""

command =\
"""move 200
turn 90 pi/2
move 50
turn -90 -pi/2
move 100
turn 90 pi/2
move 50
turn 90 pi/2
move 100
turn -90 -pi/2
move 50
turn 90 pi/2
move 300
move -300 penup
turn 90 pi/2
move 50 penup
move 50"""
command2 =\
"""ABKL a b
CEGH c f"""


command =\
"""move 200
turn 135 pi/2+a
move 350
move -250
turn -90 -pi/2
box m b"""
command2 = """ABC c 0
friction box1 ABC u"""

command = \
"""move 100
turn -90 -pi/2
box m b
turn 90 pi/2
move 100
move 20 penup
pulley
move -20 penup
turn 90 pi/2
move 100
turn -90 -pi/2
box c f
turn 90 pi/2
move 50
turn 90 pi/2
move 250"""

command2 ="""ABCD 0 0
string box1 pulleyr1
string box2 pulleyl1
string ABCD pulley1"""






command =\
"""move 100
turn -90 -pi/2
box m b
turn 90 pi/2
move 100
move 10 penup
pulley 0
move -10 penup
turn 135 pi/2+a
move 350
move -250
turn -90 -pi/2
box c f"""
command2 = """ABC 0 0
string box1 pulleyr1 s
string box2 pulleyl1 s
string ABC pulley1 2*s"""

command =\
"""box m b
turn 90 pi/2
move 100
turn -90 -pi/2
box c f
move 10 penup
turn 90 pi/2
move 50 force p"""
command2 = """string box1 box2 s"""

command =\
"""box m a
move 40 penup
move 100 force f"""
command2 =None


command =\
"""move 200
turn 135 pi/2+a
move 400
move -300
turn -90 -pi/2
box m b"""
command2 = """ABC c f"""

command =\
"""move 200 penup
wall
pulley r
move -100 penup
turn 90 pi/2
move 10 penup
turn -90 -pi/2
box m b
turn -90 -pi/2
move 20 penup
turn 90 pi/2
box c f"""

command2 = """string wall1 pulley1 2*s
string box1 pulleyl1 s
string box2 pulleyr1 s"""




command=\
"""move 200 penup
box m b"""
command2 = None

command =\
"""move 200
turn 135 pi/2+a
move 350
move -250
turn -90 -pi/2
box m b"""
command2 = """ABC c f"""
command = \
"""move 200
turn -90 -pi/2
move 50 penup
turn 90 pi/2
pulley r
move -50 penup
turn 90 pi/2
move 20 penup
turn 90 pi/2
box m b
turn -90 -pi/2
move -40 penup
turn 90 pi/2
box c f
turn -90 -pi/2
move 70 penup
turn -90 -pi/2
move 50 penup
move 20 penup
pulley 0
move -20 penup
turn 90 pi/2
move 100
turn -90 -pi/2
box v w
turn 90 pi/2
move 50
turn 90 pi/2
move 220"""
command2 = """ABCD 0 0
string pulleyr2 pulley1 2*s
string box3 pulleyl2 2*s
string box2 pulleyr1 s
string box1 pulleyl1 s
string ABCD pulley2 4*s"""

pulley = []
wall = []
box = []
force = []
linear = [[tree_form("d_0"), tree_form("d_1"), tree_form("d_250")]]
slopelinear = [tree_form("d_0")]
turn = 90
turn2 = parser.take_input("pi/2")
x = tree_form("d_0")
y = tree_form("d_-250")
prevturn = None
for item in command.split("\n"):
    
    tmp = item.split(" ")[0]
    n = None
    if tmp not in ["wall", "pulley", "box"]:
        n = int(item.split(" ")[1])
    if tmp == "move":
        orig = copy.deepcopy([x, y])
        n = "d_"+ str(n)
        
        turn = turn % 360
        m = turn // 90
        k = turn % 90
        
        slope = None
        if m == 0:
            x += tree_form(n)*parser.take_input(f"cos({k}*pi/180)")
            y += tree_form(n)*parser.take_input(f"sin({k}*pi/180)")
            slope = parser.take_input(f"tan({k}*pi/180)")
        elif m == 1:
            x += tree_form(n)*parser.take_input(f"-sin({k}*pi/180)")
            y += tree_form(n)*parser.take_input(f"cos({k}*pi/180)")
            slope = parser.take_input(f"-1/tan({k}*pi/180)")
        elif m == 2:
            x += tree_form(n)*parser.take_input(f"-cos({k}*pi/180)")
            y += tree_form(n)*parser.take_input(f"-sin({k}*pi/180)")
            slope = parser.take_input(f"tan({k}*pi/180)")
        elif m == 3:
            x += tree_form(n)*parser.take_input(f"sin({k}*pi/180)")
            y += tree_form(n)*parser.take_input(f"-cos({k}*pi/180)")
            slope = parser.take_input(f"-1/tan({k}*pi/180)")
        if (prevturn != turn and len(item.split(" ")) == 2) or (len(item.split(" ")) > 2 and item.split(" ")[2] == "force"):
            sa, sb, sc = None, None, None
            prevturn = turn
            slope = mathai.solve(mathai.replace_eq2(slope))
            if slope is None:
                sa = tree_form("d_1")
                sb = tree_form("d_0")
                sc = -x
            elif slope == tree_form("d_0"):
                sa = tree_form("d_0")
                sb = tree_form("d_1")
                sc = -y
            else:
                sa = slope
                sb = tree_form("d_-1")
                sc = y - slope * x
            
            if len(item.split(" ")) > 2 and item.split(" ")[2] == "force":
                
                force.append([[mathai.compute(orig[0]), mathai.compute(orig[1])], mathai.solve(turn2), parser.take_input(item.split(" ")[3])])
            else:
                tmp = [mathai.solve(item) for item in [sa, sb, sc]]
                if any(all(g[h] == tmp[h] for h in range(3)) for g in linear):
                   pass
                else:
                    linear.append(tmp)
                    slopelinear.append(mathai.solve(turn2))
    elif tmp == "turn":
        turn += n
        turn = turn % 360
        turn2 += parser.take_input(item.split(" ")[2])
    elif tmp == "box":
        cor = [mathai.compute(x), mathai.compute(y)]
        
        box.append([mathai.solve(turn2), parser.take_input(item.split(" ")[1]), parser.take_input(item.split(" ")[2]), turn, cor, len(linear)-1])
    elif tmp == "wall":
        wall.append([turn, [mathai.compute(x), mathai.compute(y)]])
    elif tmp == "pulley":
        pulley.append([turn, [mathai.compute(x), mathai.compute(y)], parser.take_input(item.split(" ")[-1])])
def is_point_inside_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Check if the point is between y1 and y2 vertically
        if (y1 > y) != (y2 > y):
            # Find the x-coordinate where the line crosses y = point.y
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1  # Add small epsilon to avoid div/0
            if x < x_intersect:
                inside = not inside

    return inside

def abc(lst):
    var1 = tree_form("v_0")
    var2 = tree_form("v_1")
    return TreeNode("f_eq", [var1*lst[0]+var2*lst[1]+lst[2], tree_form("d_0")])

linear2 = []
done = []

for i in range(2,len(linear)+1):
    lst = list(itertools.combinations(enumerate(linear[:i]), 2))
    lst = sorted(lst, key=lambda x: abs(x[1][0]-x[0][0]))
    for item in lst:
        item3 = item[0][0], item[1][0]
        if item3 in done:
            continue
        item = item[0][1], item[1][1]
        eq = abc(item[0]) & abc(item[1])
        eq = mathai.solve(mathai.expand2(eq))
        eq = mathai.and0(eq)
        if eq == tree_form("d_false"):
            continue
        xval, yval = None, None
        for child in eq.children:
            if "v_0" in str_form(child):
                xval = tree_form("v_0") - child.children[0]
            elif "v_1" in str_form(child):
                yval = tree_form("v_1") - child.children[0]
        xval, yval = [mathai.solve(mathai.expand2(item2)) for item2 in [xval, yval]]
        linear2.append([xval, yval])
        done.append(item3)

def get_pulley_string_points(cx, cy, radius, angle_degrees):
    angle_rad = math.radians(angle_degrees)
    
    # Left point (angle + 90°)
    left_angle = angle_rad + math.pi / 2
    left_x = cx + radius * math.cos(left_angle)
    left_y = cy + radius * math.sin(left_angle)
    
    # Right point (angle - 90°)
    right_angle = angle_rad - math.pi / 2
    right_x = cx + radius * math.cos(right_angle)
    right_y = cy + radius * math.sin(right_angle)

    return [(left_x, left_y), (right_x, right_y)]

apoint = []
for item in box:
    angle_rad= math.radians(item[-3])
    new_x, new_y = item[-2]
    d = 10
    new_x = new_x + d * math.cos(angle_rad)
    new_y = new_y + d * math.sin(angle_rad)
    apoint.append([new_x, new_y])
apoint2 = []
apoint3 = []
for item in wall:
    angle_rad= math.radians(item[0])
    new_x, new_y = item[1]
    d = 10
    new_x = new_x + d * math.cos(angle_rad)
    new_y = new_y + d * math.sin(angle_rad)
    apoint2.append([new_x, new_y])
for item in pulley:
    angle_rad= math.radians(item[0])
    new_x, new_y = item[1]
    d = -10
    new_x = new_x + d * math.cos(angle_rad)
    new_y = new_y + d * math.sin(angle_rad)
    apoint3.append([])
    apoint3[-1].append([new_x, new_y])
    tmp = get_pulley_string_points(new_x, new_y, 10, item[0])
    for item in tmp:
        apoint3[-1].append(list(item))

abcdic = {}
count = 0
if command2 is not None and command2 != "": 
    for item in command2.split("\n"):
        if item.split(" ")[0].lower()!=item.split(" ")[0]:
            
            abcdic[item.split(" ")[0]] = count
            count += 1
body_list = []
if command2 is not None and command2 != "":
    body_list = [[ord(item2)-ord("A") for item2 in item.split(" ")[0]] for item in command2.split("\n") if item.split(" ")[0].lower()!=item.split(" ")[0]] 
attach = []
direct = []
fixed = []

if command2 is not None and command2 != "":
    for item in command2.split("\n"):
        if item.split(" ")[0] == "string":
            for index, item2 in list(enumerate(item.split(" ")[1:]))[:2]:
                if index == 0:
                    direct.append([])
                    attach.append([])
                if item2.startswith("box"):
                    direct[-1].append(["box", len(body_list)+int(item2[-1:])-1])
                    attach[-1].append(apoint[int(item2[-1:])-1])
                    if index == 1:
                        direct[-1].append(parser.take_input(item.split(" ")[-1]))
                elif item2.startswith("wall"):
                    direct[-1].append(None)
                    attach[-1].append(apoint2[int(item2[-1:])-1])
                elif item2.startswith("pulleyl"):
                    direct[-1].append(["pulleyl", int(item2[-1:])-1])
                    attach[-1].append(apoint3[int(item2[-1:])-1][1])
                    if index == 1:
                        direct[-1].append(parser.take_input(item.split(" ")[-1]))
                elif item2.startswith("pulleyr"):
                    direct[-1].append(["pulleyr", int(item2[-1:])-1])
                    attach[-1].append(apoint3[int(item2[-1:])-1][2])
                    if index == 1:
                        direct[-1].append(parser.take_input(item.split(" ")[-1]))
                elif item2.startswith("pulley"):
                    direct[-1].append(["pulley", int(item2[-1:])-1])
                    if len(attach[-1])>0:
                        attach[-1].append(apoint3[int(item2[-1:])-1][0])
                    if index == 1:
                        direct[-1].append(parser.take_input(item.split(" ")[-1]))
                elif item2.lower() != item2:
                    direct[-1].append(["box", abcdic[item2]])
                    #attach[-1].append(apoint[abcdic[item2]])
                    if index == 1:
                        direct[-1].append(parser.take_input(item.split(" ")[-1]))
    
import draw8
draw8.render_turtle_commands(command, [tuple([mathai.compute(item[0]), mathai.compute(item[1]), chr(ord("A")+index)]) for index, item in enumerate(linear2)], "turtle_output.png", attach)
slopelinear2 = copy.deepcopy(slopelinear)


for i in range(len(slopelinear)):
    m = slopelinear[i]
    n = None
    if mathai.solve(mathai.expand2(m+parser.take_input("pi/2"))) == tree_form("d_0") or\
       mathai.solve(mathai.expand2(m+parser.take_input("-pi/2"))) == tree_form("d_0"):
        n = parser.take_input("pi/2")
    if m == tree_form("d_0"):
        n = tree_form("d_0")
    slopelinear[i] = n
    
def mag(cosangle, sinangle):
    return (cosangle**tree_form("d_2")+sinangle**tree_form("d_2"))**parser.take_input("1/2")

def inverse_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]

    det = a * d - b * c
    if det == tree_form("d_0"):
        raise ValueError("Matrix is singular and cannot be inverted")

    inv_det = tree_form("d_1")/det
    inverse = [
        [d * inv_det, -b * inv_det],
        [-c * inv_det, a * inv_det]
    ]
    return [[mathai.solve(mathai.expand2(mathai.fraction2(mathai.solve(mathai.expand2(item))))) for item in item2] for item2 in inverse]

def factormore(eq):
    l = len(eq.children)
    for i in range(l,1,-1):
        for item in itertools.combinations(eq.children, i):
            item = list(item)
            item = mathai.solve(mathai.summation(item))
            item2 = mathai.term_common(item)
            if item2.name == "f_mul":
                return mathai.solve(mathai.expand2(eq-item)+item2)
    return eq
def boxpoint(p):
    a, b= p
    return [[a+40, b], [a-40, b], [a+40, b+80], [a-40, b+80]]

for i in range(len(force)):
    for j in range(len(box)):
        if is_point_inside_polygon(force[i][0], boxpoint(box[j][-2])):
            force[i][0] = [j, "box"]

def is_point_on_line_general_form(a, b, c, x, y, epsilon=1e-9):
    a, b, c, x, y = [mathai.compute(item) for item in [a, b, c, x, y ]]
    return abs(a * x + b * y + c) < epsilon


x, y = mathai.solve(x), mathai.solve(y)

print(command)
if command2:
    print(command2)
print()
print("EQUATION GENERATED =")

body_list = []
body_line = []
body_mass = [x[1] for x in box]
body_acc = [x[2] for x in box] + [x[-1] for x in pulley]

if command2 is not None and command2 != "":
    command2 = "\n".join([item for item in command2.split("\n") if item.split(" ")[0] != "string"])


if command2 is not None and command2 != "":
    
    body_list = [[ord(item2)-ord("A") for item2 in item.split(" ")[0]] for item in command2.split("\n") if item.split(" ")[0].lower()!=item.split(" ")[0]] 
    body_mass = [parser.take_input(item.split(" ")[1]) for item in command2.split("\n") if item.split(" ")[0].lower()!=item.split(" ")[0]]+body_mass
    body_acc = [parser.take_input(item.split(" ")[2]) for item in command2.split("\n") if item.split(" ")[0].lower()!=item.split(" ")[0]]+body_acc

    for item in body_list:
        out = []
        for item2 in item:
            out += list(done[item2])
            out = list(set(out))
        body_line.append(out)
box_list = []
for i in range(len(box)):
    for j in range(len(body_line)):
        if box[i][-1] in body_line[j]:
            box_list.append(j)
            break
    if len(box) != len(box_list):
        box_list.append(-1)

def displace_along_slope(point, slope, distance):
    x, y = point
    
    # Direction vector from slope
    dx = 1
    dy = slope
    
    # Normalize the direction vector
    length = math.hypot(dx, dy)
    ux = dx / length
    uy = dy / length

    # Displacement
    x_new = x + distance * ux
    y_new = y + distance * uy

    return (x_new, y_new)   
def displace_anti(point, slope, vertex):
    tmp =displace_along_slope(point, slope+math.radians(90), 5)
    return is_point_inside_polygon(tmp, vertex)
    pass

def point_on_line(a, b, c, m, n, eps=1e-9):
    return abs(a * m + b * n + c) < eps

for i in range(len(box)):
    l = None
    for index, item in enumerate(linear):
        item = [mathai.compute(x) for x in item]
        if point_on_line(item[0], item[1], item[2], box[i][-2][0], box[i][-2][1]):
            l = index
            break
    box[i][-1] = l

obj_list = body_line + [[x[-1]] if x[-1] is not None else [] for x in box] + [[] for i in range(len(pulley))]
normals = {}
acc = {}

for item in itertools.permutations(list(range(len(obj_list)))+[-1], 2):
    a, b = item
    if a == -1 or b == -1:
        non_minus = a
        if non_minus == -1:
            non_minus = b
        if 0 not in obj_list[non_minus]:
            continue
        if a == -1:
            acc[non_minus] = parser.take_input("0")
            normals[(a, b)] = parser.take_input("pi/2")
        else:
            normals[(a, b)] = parser.take_input("pi/2+pi")
        normals[(b, a)] = None
        continue
    if len(obj_list[b]) == 1:
        if len(obj_list[a]) == 1:
            continue
        #if (a, b) not in normals.keys():
        normals[(a, b)] = None
    else:
        
        line = list( set(obj_list[a]) & set(obj_list[b]) )
        if line == []:
            continue
        line = line[0]
        point = [j for j, x in enumerate(done) if line in list(x)]
        two = tree_form("d_2")
        point = [  mathai.compute(linear2[point[0]][0]/two+linear2[point[1]][0]/two), mathai.compute(linear2[point[0]][1]/two+linear2[point[1]][1]/two) ]
        vertex = [[mathai.compute(linear2[x][0]), mathai.compute(linear2[x][1])] for x in body_list[b]]
        slope = None
        if slopelinear[line] is None:
            slope = math.atan(-mathai.compute(linear[line][0]/linear[line][1]))
        else:
            slope = mathai.compute(slopelinear[line])
        acc[a] = mathai.solve(slopelinear2[line]-parser.take_input("pi"))
        if (a, b) in normals.keys() and normals[(a, b)] is not None:
            pass
        if displace_anti(point, slope, vertex):
            normals[(a, b)] = mathai.solve(-parser.take_input("pi/2")+slopelinear2[line])
        else:
            normals[(a, b)] = mathai.solve(-parser.take_input("pi/2")+slopelinear2[line]-parser.take_input("pi"))
            #normals[(b, a)] = -normals[(a, b)]
for key in normals.keys():
    if normals[key] is None and (key[1], key[0]) in normals.keys() and normals[(key[1], key[0])] is not None:
        normals[key] = normals[(key[1], key[0])]+parser.take_input("pi")

for item in itertools.permutations(list(range(len(obj_list)))+[-1], 2):
    a, b= item
    '''
    if normals[(a, b)] is None or normals[(b, a)] is None:
        continue
    '''
    if (a, b) in normals.keys() and (b, a) in normals.keys() and\
       mathai.solve(mathai.expand2(normals[(a, b)] - normals[(b, a)])) not in [parser.take_input("pi"), parser.take_input("-pi")]:
        normals[(a, b)]= normals[(b, a)]+parser.take_input("pi")

body_normal = {(0, 1): parser.take_input("n"), (-1, 0): parser.take_input("d"), (-1, 1): parser.take_input("h"), (0, 2): parser.take_input("k"), (0, 3): parser.take_input("t")}

for i in range(len(obj_list)):
    if obj_list[i]  == []:
        acc[i] = parser.take_input("-pi/2")

lst = []
matheq = []
def ss(newx, newy):    
    newx = mathai.replace_eq(newx)
    newy = mathai.replace_eq(newy)
    
    newx = mathai.solve(mathai.expand2(newx))
    newy = mathai.solve(mathai.expand2(newy))

    newx = mathai.replace_eq(newx)
    newy = mathai.replace_eq(newy)

    newx = mathai.solve(mathai.expand2(newx))
    newy = mathai.solve(mathai.expand2(newy))
    if newx != parser.take_input("0"):
        matheq.append(str(newx)+" = 0")
    if newy != parser.take_input("0"):
        matheq.append(str(newy)+" = 0")
    return newx, newy

def giveneg(eq):
    if eq.name != "f_add":
        eq = TreeNode("f_add", [eq])
    neg = []
    pos = []
    for child in eq.children:
        if any(item.name[:2] == "d_" and int(item.name[2:]) <0 for item in mathai.factorgen(child)):
            neg.append(child)
        else:
            pos.append(child)
    return mathai.summation(neg), mathai.summation(pos)

def which_way_to_face_perpendicular_toward_each_other(p1, p2, facing_deg, epsilon=1e-5):
    def angle_to(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return math.degrees(math.atan2(dy, dx)) % 360

    def normalize_angle(angle):
        return angle % 360

    def angles_close(a1, a2, epsilon=1e-5):
        return abs(normalize_angle(a1 - a2)) < epsilon or abs(normalize_angle(a2 - a1)) < epsilon

    # Angle of the vector from p1 to p2
    direction_angle = angle_to(p1, p2)
    # Two perpendicular directions (facing into each other)
    desired_facing1 = direction_angle % 360
    desired_facing2 = (direction_angle - 180) % 360
    
    # Compute both clockwise and anticlockwise rotations from current facing
    cw = (facing_deg - 90) % 360
    acw = (facing_deg + 90) % 360

    turn1 = "clockwise" if angles_close(cw, desired_facing1, epsilon) else "anticlockwise"
    turn2 = "clockwise" if angles_close(cw, desired_facing2, epsilon) else "anticlockwise"

    return turn1, turn2

for i in range(len(obj_list)):
    if len(obj_list) == 1 and 0 not in obj_list[i]:
        acc[i] = parser.take_input("-pi/2")

friction = {}
if command2 is not None and command2 != "":
    for item in command2.split("\n"):
        if item.split(" ")[0] == "friction":
            out = []
            for item2 in item.split(" ")[1:3]:
                if item2.lower() != item2:
                    out.append(abcdic[item2])
                else:
                    out.append(int(item2[-1])-1 + len(body_list))
            
            friction[tuple(sorted(out))]= parser.take_input(item.split(" ")[-1])

pull = []

if len(pulley) >0:
    for i in range(len(pulley)):
        pull.append([None, [], None])
for item in direct:
    for index, item2 in enumerate(item[:2]):
        
        if item2 is not None and item2[0].startswith("pulley"):
            if item2[0][-1] not in ["l", "r"]:
                if item[1-index] is None:
                    pull[item2[1]][0] = None
                elif item[1-index][0] == "box":
                    pull[item2[1]][0] = item[1-index][1]
                elif item[1-index][0] in ["pulleyr", "pulleyl"]:
                    pull[item2[1]][0] = item[1-index][1]+len(box)+len(body_list)
                pull[item2[1]][2] = item[2]/tree_form("d_2")
            else:
                if item[1-index] is None:
                    pull[item2[1]][1].append(None)
                elif item[1-index][0] == "box":
                    pull[item2[1]][1].append(item[1-index][1])
                elif item[1-index][0] == "pulley":
                    pull[item2[1]][1].append(item[1-index][1]+len(box)+len(body_list))

for i in range(len(body_list)+len(box), len(body_acc)):
    if body_acc[i] == parser.take_input("0"):
        acc[i] = parser.take_input("0")
    else:
        acc[i] = parser.take_input("-pi/2")
for item in pull:
    eq = None
    if mathai.solve(acc[item[1][0]]-parser.take_input("-pi/2")) == tree_form("d_0") and mathai.solve(acc[item[1][1]]-parser.take_input("-pi/2")) == tree_form("d_0"):
        eq = body_acc[item[1][0]]+body_acc[item[1][1]]
    else:
        eq = body_acc[item[1][0]]-body_acc[item[1][1]]
    eq = mathai.replace_eq(eq)
    eq = mathai.solve(mathai.expand2(eq))
    eq = mathai.replace_eq(eq)
    eq = mathai.solve(mathai.expand2(eq))
    matheq.append(str(eq)+" = 0")

for i in range(len(direct)):
    if direct[i][0] is None or direct[i][1] is None or direct[i][0][0] != "box" or direct[i][1][0] != "box":
        continue
    eq = body_acc[direct[i][0][1]]-body_acc[direct[i][1][1]]
    eq = mathai.replace_eq(eq)
    eq = mathai.solve(mathai.expand2(eq))
    eq = mathai.replace_eq(eq)
    eq = mathai.solve(mathai.expand2(eq))
    matheq.append(str(eq)+" = 0")
    
for key in acc.keys():
    if body_acc[key] == parser.take_input("0"):
        
        continue
    
    ref = None

    if key >= len(box)+len(body_list):
        continue
    
    for index, item in enumerate(pull):
        if key in item[1] and item[0] >= len(box)+len(body_list):
            ref = index+len(box)+len(body_list)
            
            break
    if ref is None:
        if 0 not in obj_list[key]:
            for key2 in normals.keys():
                if key in list(key2):
                    ref = list(set(key2)-set([key]))[0]
                    break
    nx = parser.take_input("0")
    ny = parser.take_input("0")

    if ref is not None and body_acc[ref] != parser.take_input("0"):
        
        nx += tree_form("d_-1") * body_acc[ref] * acc[ref].fx("cos") * body_mass[key]
        ny += tree_form("d_-1") * body_acc[ref] * acc[ref].fx("sin") * body_mass[key]
    for index, item in enumerate(pull):
        if key in item[1]:
            if mathai.solve(acc[key]-parser.take_input("-pi/2")) == tree_form("d_0"):
                nx -= item[2] * acc[key].fx("cos")
                ny -= item[2] * acc[key].fx("sin")
            else:
                nx += item[2] * acc[key].fx("cos")
                ny += item[2] * acc[key].fx("sin")
            
    for item in direct:
        
        if item[0] is not None and item[1] is not None and item[0][0] == "box" and item[1][0] == "box" and\
           (item[0]+len(body_line) == key or item[1]+len(body_line) == key):
            a, b= which_way_to_face_perpendicular_toward_each_other(box[item[0]][-2], box[item[1]][-2], box[item[0]][-3])
            t = box[item[0]][0]
            
            if a == "clockwise":
                a = t-parser.take_input("pi/2")
            else:
                a = t+parser.take_input("pi/2")

            if b == "clockwise":
                b = t-parser.take_input("pi/2")
            else:
                b = t+parser.take_input("pi/2")

            if item[0]+len(body_line) == key:
                t = a
            else:
                t= b

            nx += item[2]*t.fx("cos")
            ny += item[2]*t.fx("sin")
            
    for key2 in normals.keys():
        if key2[1] == key and normals[key2] is not None:
            nx += normals[key2].fx("cos")*body_normal[tuple(sorted(list(key2)))]
            ny += normals[key2].fx("sin")*body_normal[tuple(sorted(list(key2)))]
    ny += parser.take_input("-g")*body_mass[key]

    for i in range(len(force)):
        if force[i][0][1] == "box" and force[i][0][0] == key-len(body_line):
            nx += force[i][1].fx("cos") * force[i][2]
            ny += force[i][1].fx("sin") * force[i][2]
    #nx, ny = ss(nx, ny)
    newx = nx * acc[key].fx("cos") + ny * acc[key].fx("sin")
    newy = nx * -acc[key].fx("sin") + ny * acc[key].fx("cos")

    
    newx -= body_acc[key] * body_mass[key]
    for key2 in friction.keys():
        if key in key2:
            newx += friction[key2] * body_normal[key2]
    
    newx, newy = ss(newx, newy)
for i in range(len(matheq)):
    matheq[i] = mathai.solve(mathai.expand2(mathai.fraction2(parser.take_input(matheq[i]))))
    print(matheq[i])
final2  = mathai.and0(TreeNode("f_and", matheq), [parser.take_input("w"), parser.take_input("r"), parser.take_input("s"), parser.take_input("n"), parser.take_input("k"), parser.take_input("d"), parser.take_input("b"), parser.take_input("f")])
print()
print("SOLUTION =")
if final2.name == "f_and":
    for item in final2.children:
        print(mathai.solve(mathai.expand2(item)))
else:
    print(final2)
