import FreeCAD
import json
import FreeCADGui as Gui
from FreeCAD import Vector
import random
# Gui.showMainWindow()
import FreeCAD as App
import Draft
print("Draft module is imported from:", Draft.__file__)
import Import
from pivy import coin
import DraftGeomUtils
import os
import inspect
import math
import PySide
# base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# hanomi_ai_dir = os.path.join(base_dir, 'hanomi-AI')

# sys.path.append(hanomi_ai_dir)
from config import FREECAD_WIN
from PySide import QtCore,QtGui,QtSvg
print("Pyside module is imported from:", PySide.__file__)
from FreeCAD import Units
# Assuming 'hanomi-AI' is at the same level as 'Quotes' directory

try:
    from PySide import QtGui
except ImportError:
    FreeCAD.Console.PrintMessage("Error: Python-pyside package must be installed on your system to use the Geometric Dimensioning & Tolerancing module.")

if FreeCAD.GuiUp:
    gui = True
else:
    FreeCAD.Console.PrintMessage("FreeCAD Gui not present. GDT module will have some features disabled.")
    gui = False
# __dir__ = os.path.dirname(__file__)
# iconPath = os.path.join( __dir__, 'Gui','Resources', 'icons' )
MBD_DIR = os.path.abspath(os.path.dirname(__file__))
dictionaryAnnotation=[]
for i in range(1,100):
    dictionaryAnnotation.append('Annotation'+str(i))

def load_step_file(filename):
    print('starting loading')
    doc = App.newDocument('test')
    Import.insert(filename, doc.Name)
    doc.recompute()
    print('end loading')
    return doc.Objects[0]

def extract_face_data():
    ''' Extract face names and references from the imported STEP objects '''
    face_data = {}
    for obj in FreeCAD.ActiveDocument.Objects:
        if obj.TypeId == "Part::Feature":
            for i, face in enumerate(obj.Shape.Faces):
                face_name = "Face" + str(i + 1)
                face_data[face_name] = face
    return face_data

# Load a STEP file (replace 'path_to_file.step' with the actual file path)
# step_file_path = 'path_to_file.step'
# load_step_file(step_file_path)

# Extract face information from the loaded STEP file
# face_info = extract_face_data()

# model_object = {
#     'annotation_frame': {
#         'label': 'AP1',
#         'annotation_plane': {
#             'p1': {'x': -0.00, 'y': 126.92, 'z': 25.88},
#             'faces': 'Face4',
#             'offset': 10
#         }
#     },
#     'datum_system': {
#         'label': 'DS1',
#         'primary': 'Face4',
#         'secondary': None,
#         'tertiary': None
#     },
#     'geometric_tolerance': {
#         'label': 'GT1',
#         'Characteristic': 'Straightness',
#         'Circumference': 'true',
#         'Tolerance_value': 1,
#         'DS': 'DS1',
#         'Feature-Control-Frame': 'Maximum material condition',
#         'p1': {'x': 46.12, 'y': 148.28, 'z': 35.88}
#     }
# }

class _GDTObject:
    "The base class for GDT objects"
    def __init__(self,obj,tp="Unknown"):
        '''Add some custom properties to our GDT feature'''
        obj.Proxy = self
        self.Type = tp

    def __getstate__(self):
        return self.Type

    def __setstate__(self,state):
        if state:
            self.Type = state

    def execute(self,obj):
        '''Do something when doing a recomputation, this method is mandatory'''
        pass


class _ViewProviderGDT:
    "Simplified ViewProvider for creating annotations in FreeCAD GUI"

    def __init__(self, vobj):
        '''Initialize the view provider and set it as the proxy for the actual view provider'''
        vobj.Proxy = self
        self.Object = vobj.Object

    def attach(self, vobj):
        '''Setup the visual representation of the view provider. This is mandatory.'''
        self.Object = vobj.Object
        # Initialize any necessary nodes for Coin3D (Pivy) scene graph here

    def getIcon(self):
        '''Return the icon for the tree view. This can be customized as needed.'''
        return "path/to/icon.svg"  # Provide the path to your icon here

    def updateData(self, obj, prop):
        '''Update visual representation in response to changes in object data.'''
        # Implement any necessary updates to the visual representation here
        pass

    def onChanged(self, vobj, prop):
        '''Respond to changes in the ViewObject properties, such as visibility or color.'''
        # Implement any necessary response to property changes here
        pass


class _Annotation(_GDTObject):
    "The GDT Annotation object"
    def __init__(self, obj):
        _GDTObject.__init__(self,obj,"Annotation")
        obj.addProperty("App::PropertyLinkSubList","faces","GDT","Linked faces of the object")
        obj.addProperty("App::PropertyLink","AP","GDT","Annotation plane used")
        obj.addProperty("App::PropertyLink","DF","GDT","Text").DF=None
        obj.addProperty("App::PropertyLinkList","GT","GDT","Text").GT=[]
        obj.addProperty("App::PropertyVectorDistance","p1","GDT","Start point")
        # can calculate the direction simply using the normal of the face pointing outwards (hardcoded for now)
        obj.addProperty("App::PropertyVector","Direction","GDT","The normal direction of your annotation plane")
        # can calculate random point which lies outside the face on the annotation plane (hardcoded for now)
        obj.addProperty("App::PropertyVector","selectedPoint","GDT","Selected point to where plot the annotation")
        obj.addProperty("App::PropertyBool","spBool","GDT","Boolean to confirm that a selected point exists").spBool = False
        obj.addProperty("App::PropertyBool","circumferenceBool","GDT","Boolean to determine if this annotation is over a circumference").circumferenceBool = False
        obj.addProperty("App::PropertyFloat","diameter","GDT","Diameter")
        obj.addProperty("App::PropertyFloat","toleranceDiameter","GDT","Diameter tolerance (Plus-minus)")

    def execute(self, fp):
        '''"Print a short message when doing a recomputation, this method is mandatory" '''
        # FreeCAD.Console.PrintMessage('Executed\n')
        # print('150 faces: ', fp.faces)
        auxP1 = fp.p1
        if fp.circumferenceBool:
            vertexes = fp.faces[0][0].Shape.getElement(fp.faces[0][1][0]).Vertexes
            fp.p1 = vertexes[0].Point if vertexes[0].Point.z > vertexes[1].Point.z else vertexes[1].Point
            fp.Direction = fp.AP.Direction
        else:
            # print('156 faces: ', fp.faces)
            fp.p1 = (fp.faces[0][0].Shape.getElement(fp.faces[0][1][0]).CenterOfMass).projectToPlane(fp.AP.PointWithOffset, fp.AP.Direction)
            fp.Direction = fp.faces[0][0].Shape.getElement(fp.faces[0][1][0]).normalAt(0,0)
            # print('160 worked')
        diff = fp.p1-auxP1
        if fp.spBool:
            fp.selectedPoint = fp.selectedPoint + diff

import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("C:\\Users\\Drishti Bhasin\\Documents\\FreeCAD 0.21\\Mod\\FreeCAD-GDT\\mbd_test.py"))))
import Part
import random
print("Part module is imported from:", Part.__file__)
print( Part.LineSegment)
def find_random_point_on_edge(edge):
    # Get the start and end points of the edge
    start_point = edge.Vertexes[0].Point
    end_point = edge.Vertexes[1].Point
    
    # Generate a random value between 0 and 1
    t = random.random()
    
    # Interpolate between the start and end points using the random value t
    x = start_point.x + t * (end_point.x - start_point.x)
    y = start_point.y + t * (end_point.y - start_point.y)
    z = start_point.z + t * (end_point.z - start_point.z)
    
    # Return the calculated point
    return (x, y, z)

def create_parallel_edge(original_edge, distance):
    # Get the start and end points of the original edge
    start_point = original_edge.Vertexes[0].Point
    end_point = original_edge.Vertexes[1].Point

    # Calculate the direction vector of the edge
    direction_vector = end_point.sub(start_point)

    # Normalize the direction vector to get a unit vector
    direction_vector.normalize()

    # Calculate a perpendicular vector to the direction vector for the 2D case
    perpendicular_vector = FreeCAD.Vector(-direction_vector.y, direction_vector.x, 0)

    # Multiply the perpendicular vector by the desired distance
    offset_vector = perpendicular_vector.multiply(distance)

    # Calculate the start and end points of the parallel edge
    parallel_start = start_point.add(offset_vector)
    parallel_end = end_point.add(offset_vector)

    # Create the parallel edge using the Draft module or Part module
    parallel_edge = Part.makeLine(parallel_start, parallel_end)

    return parallel_edge


def is_parallel_and_equal(edge1, edge2):
    # Calculate direction vectors of the edges
    dir1 = edge1.Vertexes[1].Point.sub(edge1.Vertexes[0].Point).normalize()
    dir2 = edge2.Vertexes[1].Point.sub(edge2.Vertexes[0].Point).normalize()
    # Check if edges are parallel (cross product is close to zero)
    if dir1.cross(dir2).Length < 1e-6:
        # Check if lengths are approximately equal
        return abs(edge1.Length - edge2.Length) < 1e-6
    return False

def is_full_circle(edge):
    try:
        if edge.Curve.__class__.__name__ != 'Circle':
            return False
    except:
        pass 
    # Check if the edge represents a full circle (angle span close to 2*pi)
    return abs(edge.LastParameter - edge.FirstParameter - 2 * math.pi) < 1e-6

def makeDimension(model):
    straight_edges = []
    for edge in model.Shape.Edges:
        try:
            if edge.Curve.__class__.__name__ == 'Line':
                straight_edges.append(edge)
        except TypeError:
            print(dir(edge))

    # Sort edges by length in descending order
    straight_edges.sort(key=lambda e: e.Length, reverse=True)

    # Filter out parallel edges of equal length
    filtered_edges = []
    for edge in straight_edges:
        if not any(is_parallel_and_equal(edge, e) for e in filtered_edges):
            filtered_edges.append(edge)

    # Take the top 20% of the longest, unique straight edges
    longest_edges = filtered_edges[:int(len(filtered_edges) * 0.5)]

    # Dimension the longest straight edges
    for edge in longest_edges:
        new_edge = create_parallel_edge(edge, 20)
        p1 = edge.Vertexes[0].Point
        p2 = edge.Vertexes[1].Point
        dimline = find_random_point_on_edge(new_edge)
        Draft.makeDimension(p1, p2, dimline)

    # Dimension complete circular edges
    for i, edge in enumerate(model.Shape.Edges, start=1):
        if is_full_circle(edge):
            Draft.makeDimension(model, i-1, 'diameter', None)
    App.ActiveDocument.recompute()

def makeAnnotation(faces, AP, DF=None, GT=[], selected_point=None, modify=False, Object=None, diameter = 0.0, toleranceSelect = True, toleranceDiameter = 0.0, lowLimit = 0.0, highLimit = 0.0):
    ''' Explanation
    '''
    # print('*************1249*******************')
    # print(faces,AP, DF,  GT, modify, Object, diameter, toleranceSelect, toleranceDiameter, lowLimit, highLimit)
    obj = FreeCAD.ActiveDocument.addObject("App::DocumentObjectGroupPython",dictionaryAnnotation[len(getAllAnnotationObjects())])
    _Annotation(obj)
    group = FreeCAD.ActiveDocument.getObject("GDT")
    # group.addObject(obj)
    # print(faces, '173')
    obj.faces = faces
    print
    obj.AP = AP
    if obj.circumferenceBool:
        print('285: ', obj.faces[0], obj.faces[0][0])
        vertexex = FreeCAD.ActiveDocument.getObject(obj.faces[0][0].Name).Shape.getElement(obj.faces[0][1]).Vertexes
        index = [l.Point.z for l in vertexex].index(max([l.Point.z for l in vertexex]))
        obj.p1 = vertexex[index].Point
        obj.Direction = obj.AP.Direction

    else:
        # print('291: ', obj.faces, obj.faces[0], obj.faces[0][1][0], obj.faces[0][0].Shape.getElement(obj.faces[0][1][0]))
        print('294', obj.faces)
        obj.p1 = (obj.faces[0][0].Shape.getElement(obj.faces[0][1][0]).CenterOfMass).projectToPlane(obj.AP.PointWithOffset, obj.AP.Direction)
        obj.Direction = obj.faces[0][0].Shape.getElement(obj.faces[0][1][0]).normalAt(0,0)
    
    # Calculate the maximum width of the part
    # Calculate the maximum width of the part
    maxWidth = getMaxWidth(obj.faces[0][0])

    # Determine the bounding box of the entire part
    boundingBox = FreeCAD.ActiveDocument.getObject(obj.faces[0][0].Name).Shape.BoundBox

    # Inflate the bounding box to create the imaginary cuboid
    cuboidMin = FreeCAD.Vector(boundingBox.XMin, boundingBox.YMin, boundingBox.ZMin) - FreeCAD.Vector(maxWidth/2, maxWidth/2, maxWidth/2)
    cuboidMax = FreeCAD.Vector(boundingBox.XMax, boundingBox.YMax, boundingBox.ZMax) + FreeCAD.Vector(maxWidth/2, maxWidth/2, maxWidth/2)

    # Function to ensure the point is outside the cuboid
    def ensureOutsideCuboid(point, cuboidMin, cuboidMax):
        for i in range(3):  # Check x, y, z coordinates
            if cuboidMin[i] < point[i] < cuboidMax[i]:
                # If point is inside the cuboid, adjust it to be just outside
                if abs(point[i] - cuboidMin[i]) < abs(point[i] - cuboidMax[i]):
                    point[i] = cuboidMin[i] - 0.01  # Slightly outside the cuboid
                else:
                    point[i] = cuboidMax[i] + 0.01
        return point

    # Assuming face_normal and face_center are defined as in your original code...
    face_normal =  obj.faces[0][0].Shape.getElement(obj.faces[0][1][0]).normalAt(0,0)
    face_center = obj.faces[0][0].Shape.getElement(obj.faces[0][1][0]).CenterOfMass

    # Calculate a point along the face normal
    random_plane_point = face_center + face_normal.normalize() * maxWidth

    # Ensure the selected point is outside the cuboid
    selected_point = ensureOutsideCuboid(random_plane_point, cuboidMin, cuboidMax)

    # Set the selected point
    # if selected_point:
    #     obj.selectedPoint.x = selected_point[0]
    #     obj.selectedPoint.y = selected_point[1]
    #     obj.selectedPoint.z = selected_point[2]
    # else:
    obj.selectedPoint = selected_point
    obj.spBool = True
    obj.DF = DF
    obj.GT = GT
    obj.diameter = diameter
    obj.toleranceDiameter = toleranceDiameter
    _ViewProviderAnnotation(obj.ViewObject)
    for l in getAllAnnotationObjects():
        l.touch()
    FreeCAD.ActiveDocument.recompute()
    return obj

def getMaxWidth(AP):
    # Placeholder function to calculate the maximum width of the part
    # This might involve iterating over the vertices of the part to find the maximum spread
    return max(AP.Shape.BoundBox.XLength, AP.Shape.BoundBox.YLength, AP.Shape.BoundBox.ZLength)

def makeAnnotationPlane(model_object, model, face_data, key):
    ''' Create an AnnotationPlane using predefined values from the model_object dictionary. '''
    Name = key
    Offset = model_object['annotation_frame'][Name]['annotation_plane']['offset']

    # if len(getAllAnnotationPlaneObjects()) == 0:
    group = FreeCAD.ActiveDocument.addObject("App::DocumentObjectGroupPython", "GDT")
    _GDTObject(group)
    _ViewProviderGDT(group.ViewObject)
    # else:
    #     group = FreeCAD.ActiveDocument.getObject("GDT")

    obj = FreeCAD.ActiveDocument.addObject("App::FeaturePython", "AnnotationPlane")
    _AnnotationPlane(obj, [model_object['annotation_frame'][Name]['annotation_plane']['faces']], model, Offset, face_data)
    if gui:
        _ViewProviderAnnotationPlane(obj.ViewObject)
    obj.Label = Name
    group.addObject(obj)

    for l in getAllAnnotationObjects():
        l.touch()
    FreeCAD.ActiveDocument.recompute()

    return obj

def makeGeometricTolerance(model_object, model, label, DF=None, AP=None):
    ''' Create a GeometricTolerance object using predefined values from the model_object dictionary. '''
    geometric_tolerance_data = model_object['geometric_tolerance'][label]
    print('382', geometric_tolerance_data)
    Name = label
    Characteristic = geometric_tolerance_data['Characteristic']
    if not Characteristic:
        Characteristic = ''
        CharacteristicIcon = ''
    else:
        CharacteristicIcon = os.path.join(MBD_DIR, 'icons', 'Characteristic', Characteristic.lower() + '.svg')
    Circumference = geometric_tolerance_data['Circumference']
    if not Circumference:
        Circumference = False
    ToleranceValue = geometric_tolerance_data['Tolerance_value']
    print('393: ', ToleranceValue)
    if not ToleranceValue:
        ToleranceValue = 0
    FeatureControlFrame = geometric_tolerance_data['Feature-Control-Frame']
    if FeatureControlFrame:
        FeatureControlFrameIcon = os.path.join(MBD_DIR, 'icons', 'FeatureControlFrame', FeatureControlFrame.replace(" ", "") , '.svg')
    else:
        FeatureControlFrame = ''
        FeatureControlFrameIcon =''
    DS = geometric_tolerance_data['ds']

    obj = FreeCAD.ActiveDocument.addObject("App::FeaturePython", "GeometricTolerance")
    _GeometricTolerance(obj)
    # if gui:
    _ViewProviderGeometricTolerance(obj.ViewObject)
    obj.Label = Name
    obj.Characteristic = Characteristic
    obj.CharacteristicIcon = CharacteristicIcon
    obj.Circumference = Circumference
    obj.ToleranceValue = ToleranceValue
    print('413:', obj.ToleranceValue)
    obj.FeatureControlFrame = FeatureControlFrame
    obj.FeatureControlFrameIcon = FeatureControlFrameIcon
    # print(obj.FeatureControlFrameIcon)
    obj.DS = DS  # This might require adjustment based on how DS is structured in your model_object

    group = FreeCAD.ActiveDocument.getObject("GDT")
    # group.addObject(obj)

    # Assuming that the annotation related to this geometric tolerance is defined in the model_object
    # You may need to adjust this part depending on how your model_object and related functions are structured
    # annotation_data = model_object.get('annotation_frame', None)
    # if annotation_data:
    # print(label, '299' )
    # selected_point = model_object['geometric_tolerance'][label]['selected Point']
    makeAnnotation((model,geometric_tolerance_data['faces']), model_object['annotation_frame'][AP]['object'], DF=DF, GT=obj) # Assuming makeAnnotation is adapted to use model_object

    for l in getAllAnnotationObjects():
        l.touch()
    FreeCAD.ActiveDocument.recompute()

    return obj

def makeDatumSystem(dfs, label):
    print('443: ', dfs, label)
    ''' Create a DatumSystem object using predefined values from the model_object dictionary. '''
    Primary = dfs.get('Primary Datum', None)
    Secondary = dfs.get('Secondary Datum', None)  # Ensure this is an object reference in FreeCAD
    Tertiary = dfs.get('Tertiary Datum', None)  # Ensure this is an object reference in FreeCAD

    obj = FreeCAD.ActiveDocument.addObject("App::FeaturePython", "DatumSystem")
    _DatumSystem(obj)
    if gui:
        _ViewProviderDatumSystem(obj.ViewObject)
    obj.Label = label
    obj.Primary = Primary
    obj.Secondary = Secondary
    obj.Tertiary = Tertiary

    group = FreeCAD.ActiveDocument.getObject("GDT")
    
    # Check and remove the object from any existing group before adding to the new one
    for grp in FreeCAD.ActiveDocument.Objects:
        if hasattr(grp, 'Group') and obj in grp.Group:
            grp.removeObject(obj)  # Remove the object from the current group

    # group.addObject(obj)  # Add the object to the new group

    for l in getAllAnnotationObjects():
        l.touch()
    FreeCAD.ActiveDocument.recompute()

    return obj


class _DatumFeature(_GDTObject):
    "The GDT DatumFeature object"
    def __init__(self, obj):
        _GDTObject.__init__(self, obj, "DatumFeature")

    def execute(self, obj):
        '''Do something when doing a recomputation, this method is mandatory'''
        pass

def makeDatumFeature(name):
    ''' Explanation
    '''
    obj = FreeCAD.ActiveDocument.addObject("App::FeaturePython","DatumFeature")
    _DatumFeature(obj)
    # if gui:
    #     _ViewProviderDatumFeature(obj.ViewObject)
    obj.Label = name
    group = FreeCAD.ActiveDocument.getObject("GDT")
    # group.addObject(obj)
    # AnnotationObj = getAnnotationObj(ContainerOfData)
    # if AnnotationObj == None:
    # makeAnnotation((model, [model_object['datum_feature'][name][key]['faces']]), model_object['annotation_frame']['AP1']['object'], DF=obj, GT=[])
    # else:
        # faces = AnnotationObj.faces
        # AP = AnnotationObj.AP
        # GT = AnnotationObj.GT
        # diameter = AnnotationObj.diameter
        # toleranceDiameter = AnnotationObj.toleranceDiameter
        # group = makeAnnotation(faces, AP, DF=obj, GT=GT, modify = True, Object = AnnotationObj, diameter=diameter, toleranceDiameter=toleranceDiameter)
        # group.addObject(obj)
    # modifying each annotation plane to show which plane needed recomputation or not
    for l in getAllAnnotationObjects():
        l.touch()
    FreeCAD.ActiveDocument.recompute()
    return obj

############ utilities ##################


def hideGrid():
    if hasattr(FreeCADGui,"Snapper") and getParam("alwaysShowGrid") == False:
        if FreeCADGui.Snapper.grid:
            if FreeCADGui.Snapper.grid.Visible:
                FreeCADGui.Snapper.grid.off()
                FreeCADGui.Snapper.forceGridOff=True

class _ViewProviderDatumSystem(_ViewProviderGDT):
    "A View Provider for the GDT DatumSystem object"
    def __init__(self, obj):
        _ViewProviderGDT.__init__(self,obj)

    def updateData(self, obj, prop):
        "called when the base object is changed"
        if prop in ["Primary","Secondary","Tertiary"]:
            textName = obj.Label.split(":")[0]
            if obj.Primary != None:
                textName+=': '+obj.Primary.Label
                if obj.Secondary != None:
                    textName+=' | '+obj.Secondary.Label
                    if obj.Tertiary != None:
                        textName+=' | '+obj.Tertiary.Label
            obj.Label = textName

    def getIcon(self):
        return(os.path.join(MBD_DIR, 'icons', 'datumSystem.svg'))

class _ViewProviderAnnotation(_ViewProviderGDT):
    "A View Provider for the GDT Annotation object"
    def __init__(self, obj):
        obj.addProperty("App::PropertyFloat","LineWidth","GDT","Line width").LineWidth = getLineWidth()
        obj.addProperty("App::PropertyColor","LineColor","GDT","Line color").LineColor = getRGBLine()
        obj.addProperty("App::PropertyFloat","LineScale","GDT","Line scale").LineScale = getParam("lineScale",2.5)
        obj.addProperty("App::PropertyLength","FontSize","GDT","Font size").FontSize = 5.0
        obj.addProperty("App::PropertyString","FontName","GDT","Font name").FontName = getTextFamily()
        obj.addProperty("App::PropertyColor","FontColor","GDT","Font color").FontColor = getRGBText()
        obj.addProperty("App::PropertyInteger","Decimals","GDT","The number of decimals to show").Decimals =  FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Units").GetInt("Decimals",2)
        obj.addProperty("App::PropertyBool","ShowUnit","GDT","Show the unit suffix").ShowUnit = getParam("showUnit",True)
        _ViewProviderGDT.__init__(self,obj)

    def attach(self, obj):
        "called on object creation"
        # print('ATTACH WAS CALLED')
        self.node = coin.SoGroup()
        self.node3d = coin.SoGroup()
        self.lineColor = coin.SoBaseColor()
        self.textColor = coin.SoBaseColor()

        self.data = coin.SoCoordinate3()
        self.data.point.isDeleteValuesEnabled()
        self.lines = coin.SoIndexedLineSet()

        selectionNode = coin.SoType.fromName("SoFCSelection").createInstance()
        selectionNode.documentName.setValue(FreeCAD.ActiveDocument.Name)
        selectionNode.objectName.setValue(obj.Object.Name) # here obj is the ViewObject, we need its associated App Object
        selectionNode.subElementName.setValue("Lines")
        selectionNode.addChild(self.lines)
        self.font = coin.SoFont()
        self.font3d = coin.SoFont()
        self.font.size.setValue(30)
        self.font3d.size.setValue(30)
        self.textDF = coin.SoAsciiText()
        self.textDF3d = coin.SoText2()
        self.textDF.string = "" # some versions of coin crash if string is not set
        self.textDF3d.string = ""
        self.textDFpos = coin.SoTransform()
        self.textDF.justification = self.textDF3d.justification = coin.SoAsciiText.CENTER
        labelDF = coin.SoSeparator()
        labelDF.addChild(self.textDFpos)
        labelDF.addChild(self.textColor)
        labelDF.addChild(self.font)
        labelDF.addChild(self.textDF)
        labelDF3d = coin.SoSeparator()
        labelDF3d.addChild(self.textDFpos)
        labelDF3d.addChild(self.textColor)
        labelDF3d.addChild(self.font3d)
        labelDF3d.addChild(self.textDF3d)

        self.textGT = []
        self.textGT3d = []
        self.textGTpos = []
        self.svg = []
        self.svgPos = []
        self.points = []
        self.face = []
        self.textureTransform = []
        for i in range(20):
            self.textGT.append(coin.SoAsciiText())
            self.textGT3d.append(coin.SoText2())
            self.textGT[i].string = ""
            self.textGT3d[i].string = ""
            self.textGTpos.append(coin.SoTransform())
            self.textGT[i].justification = self.textGT3d[i].justification = coin.SoAsciiText.CENTER
            labelGT = coin.SoSeparator()
            labelGT.addChild(self.textGTpos[i])
            labelGT.addChild(self.textColor)
            labelGT.addChild(self.font)
            labelGT.addChild(self.textGT[i])
            labelGT3d = coin.SoSeparator()
            labelGT3d.addChild(self.textGTpos[i])
            labelGT3d.addChild(self.textColor)
            labelGT3d.addChild(self.font3d)
            labelGT3d.addChild(self.textGT3d[i])
            self.svg.append(coin.SoTexture2())
            self.face.append(coin.SoFaceSet())
            self.textureTransform.append(coin.SoTexture2Transform())
            self.svgPos.append(coin.SoTextureCoordinatePlane())
            self.face[i].numVertices = 0
            self.points.append(coin.SoVRMLCoordinate())
            image = coin.SoSeparator()
            image.addChild(self.svg[i])
            image.addChild(self.textureTransform[i])
            image.addChild(self.svgPos[i])
            image.addChild(self.points[i])
            image.addChild(self.face[i])
            self.node.addChild(labelGT)
            self.node3d.addChild(labelGT3d)
            self.node.addChild(image)
            self.node3d.addChild(image)

        self.drawstyle = coin.SoDrawStyle()
        self.drawstyle.style = coin.SoDrawStyle.LINES
        self.drawstyle.lineWidth = 10.0

        self.node.addChild(labelDF)
        self.node.addChild(self.drawstyle)
        self.node.addChild(self.lineColor)
        self.node.addChild(self.data)
        self.node.addChild(self.lines)
        self.node.addChild(selectionNode)
        obj.addDisplayMode(self.node,"2D")

        self.node3d.addChild(labelDF3d)
        self.node3d.addChild(self.lineColor)
        self.node3d.addChild(self.data)
        self.node3d.addChild(self.lines)
        self.node3d.addChild(selectionNode)
        obj.addDisplayMode(self.node3d,"3D")
        self.onChanged(obj,"LineColor")
        self.onChanged(obj,"LineWidth")
        self.onChanged(obj,"FontSize")
        self.onChanged(obj,"FontName")
        self.onChanged(obj,"FontColor")

    def updateData(self, fp, prop):
        # print('UPDATE DATA IS CALLED')
        "If a property of the handled feature has changed we have the chance to handle this here"
        # fp is the handled feature, prop is the name of the property that has changed
        if prop in "selectedPoint" and hasattr(fp.ViewObject,"Decimals") and hasattr(fp.ViewObject,"ShowUnit") and fp.spBool:
            points, segments = getPointsToPlot(fp)
            # print(str(points))
            # print(str(segments))
            self.data.point.setNum(len(points))
            cnt=0
            for p in points:
                self.data.point.set1Value(cnt,p.x,p.y,p.z)
                cnt=cnt+1
            self.lines.coordIndex.setNum(len(segments))
            self.lines.coordIndex.setValues(0,len(segments),segments)
            plotStrings(self, fp, points)
        if prop in "faces" and fp.faces != []:
            fp.circumferenceBool = True if (True in [l.Closed for l in fp.faces[0][0].Shape.getElement(fp.faces[0][1][0]).Edges] and len(fp.faces[0][0].Shape.getElement(fp.faces[0][1][0]).Vertexes) == 2) else False

    # def doubleClicked(self,obj):
    #     try:
    #         select(self.Object)
    #     except:
    #         select(obj.Object)

    def getDisplayModes(self,obj):
        "Return a list of display modes."
        modes=[]
        modes.append("2D")
        modes.append("3D")
        return modes

    def getDefaultDisplayMode(self):
        "Return the name of the default display mode. It must be defined in getDisplayModes."
        return "2D"

    def setDisplayMode(self,mode):
        return mode
class _ViewProviderGeometricTolerance(_ViewProviderGDT):
    "A View Provider for the GDT GeometricTolerance object"
    def __init__(self, obj):
        _ViewProviderGDT.__init__(self,obj)

    def getIcon(self):
        icon = self.Object.CharacteristicIcon
        return icon

    def attach(self, obj):
        "Simplified attachment method without interactive elements"
        self.node = coin.SoGroup()
        self.lineColor = coin.SoBaseColor()
        self.textColor = coin.SoBaseColor()

        self.data = coin.SoCoordinate3()
        self.lines = coin.SoIndexedLineSet()

        self.font = coin.SoFont()
        self.font.size.setValue(30)
        self.textDF = coin.SoAsciiText()
        self.textDF.string = ""  # Initial empty string

        self.drawstyle = coin.SoDrawStyle()
        self.drawstyle.style = coin.SoDrawStyle.LINES
        self.drawstyle.lineWidth = 5.0

        self.node.addChild(self.drawstyle)
        self.node.addChild(self.lineColor)
        self.node.addChild(self.data)
        self.node.addChild(self.lines)
        obj.addDisplayMode(self.node, "2D")

        # Simplified font and color settings, removing interactive parts
        self.font.size = obj.FontSize.Value
        self.lineColor.rgb.setValue(obj.LineColor[0], obj.LineColor[1], obj.LineColor[2])
        self.textColor.rgb.setValue(obj.FontColor[0], obj.FontColor[1], obj.FontColor[2])
        self.drawstyle.lineWidth = obj.LineWidth

    def getIcon(self):
        return(os.path.join(MBD_DIR, 'icons', 'annotation.svg'))

class _ViewProviderAnnotationPlane(_ViewProviderGDT):
    "A View Provider for the GDT AnnotationPlane object"
    def __init__(self, obj):
        _ViewProviderGDT.__init__(self,obj)

    def updateData(self, obj, prop):
        "called when the base object is changed"
        if prop in ["Point","Direction","Offset"]:
            obj.PointWithOffset = obj.p1 + obj.Direction * obj.Offset

    def getIcon(self):
        return(os.path.join(MBD_DIR, 'icons', 'annotationPlane.svg'))


class _GeometricTolerance(_GDTObject):
    "The GDT GeometricTolerance object"
    def __init__(self, obj):
        _GDTObject.__init__(self, obj, "GeometricTolerance")
        obj.addProperty("App::PropertyString", "Characteristic", "GDT", "Characteristic of the geometric tolerance")
        obj.addProperty("App::PropertyString", "CharacteristicIcon", "GDT", "Characteristic icon path of the geometric tolerance")
        obj.addProperty("App::PropertyBool", "Circumference", "GDT", "Indicates whether the tolerance applies to a given diameter")
        obj.addProperty("App::PropertyFloat", "ToleranceValue", "GDT", "Tolerance value of the geometric tolerance")
        obj.addProperty("App::PropertyString", "FeatureControlFrame", "GDT", "Feature control frame of the geometric tolerance")
        obj.addProperty("App::PropertyString", "FeatureControlFrameIcon", "GDT", "Feature control frame icon path of the geometric tolerance")
        obj.addProperty("App::PropertyLink", "DS", "GDT", "Datum system used")

class _DatumFeature(_GDTObject):
    "The GDT DatumFeature object"
    def __init__(self, obj):
        _GDTObject.__init__(self, obj, "DatumFeature")

    def execute(self, obj):
        '''Do something when doing a recomputation, this method is mandatory'''
        pass

class _ViewProviderDatumFeature(_ViewProviderGDT):
    "A View Provider for the GDT DatumFeature object"
    def __init__(self, obj):
        _ViewProviderGDT.__init__(self,obj)

    def getIcon(self):
        return(os.path.join(MBD_DIR, 'icons', 'datumFeature.svg'))

class _DatumSystem(_GDTObject):
    "The GDT DatumSystem object"
    def __init__(self, obj):
        _GDTObject.__init__(self,obj,"DatumSystem")
        obj.addProperty("App::PropertyLink","Primary","GDT","Primary datum feature used")
        obj.addProperty("App::PropertyLink","Secondary","GDT","Secondary datum feature used")
        obj.addProperty("App::PropertyLink","Tertiary","GDT","Tertiary datum feature used")

class _AnnotationPlane(_GDTObject):
    "The simplified GDT AnnotationPlane object without user interaction dependencies"
    def __init__(self, obj, faces, model, offset, face_data):
        _GDTObject.__init__(self, obj, "AnnotationPlane")
        self.faces = faces  # Assuming faces is a predefined parameter passed to the constructor
        self.offset = offset  # Assuming offset is a predefined parameter passed to the constructor

        obj.addProperty("App::PropertyFloat", "Offset", "GDT", "The offset value to apply to this annotation plane").Offset = self.offset
        # Assuming faces information is provided as a parameter, eliminating the need for getSelectionEx()
        obj.addProperty("App::PropertyLinkSub","faces","GDT","Linked face of the object").faces = (model, faces)
        obj.addProperty("App::PropertyVectorDistance", "p1", "GDT", "Center point of Grid").p1 = face_data[faces[0]].CenterOfMass
        obj.addProperty("App::PropertyVector", "Direction", "GDT", "The normal direction of this annotation plane").Direction = face_data[faces[0]].normalAt(0,0)
        obj.addProperty("App::PropertyVectorDistance", "PointWithOffset", "GDT", "Center point of Grid with offset applied")

    def onChanged(self, vobj, prop):
        # Simplified to avoid any GUI-specific actions
        pass

    def execute(self, fp):
        '''"Print a short message when doing a recomputation, this method is mandatory" '''
        # Assuming the faces information does not change, simplifying the execute function
        pass

def getAnnotationObj(obj):
    List = getAllAnnotationObjects()
    for l in List:
        if l.faces == obj.faces:
            return l
    return None

def getAnnotationWithDF(obj):
    List = getAllAnnotationObjects()
    for l in List:
        if l.DF == obj:
            return l
    return None

def getAnnotationWithGT(obj):
    List = getAllAnnotationObjects()
    for l in List:
        for gt in l.GT:
            if gt == obj:
                return l
    return None

def getType(obj):
    "getType(object): returns the GDT type of the given object"
    if not obj:
        return None
    if "Proxy" in obj.PropertiesList:
        if hasattr(obj.Proxy,"Type"):
            return obj.Proxy.Type
    return "Unknown"

def getObjectsOfType(typeList):
    "getObjectsOfType(string): returns a list of objects of the given type"
    listObjectsOfType = []
    objs = FreeCAD.ActiveDocument.Objects
    if not isinstance(typeList,list):
        typeList = [typeList]
    for obj in objs:
        for typ in typeList:
            if typ == getType(obj):
                listObjectsOfType.append(obj)
    return listObjectsOfType

def getAllAnnotationPlaneObjects():
    "getAllAnnotationPlaneObjects(): returns a list of annotation plane objects"
    return getObjectsOfType("AnnotationPlane")

def getAllDatumFeatureObjects():
    "getAllDatumFeatureObjects(): returns a list of datum feature objects"
    return getObjectsOfType("DatumFeature")

def getAllDatumSystemObjects():
    "getAllDatumSystemObjects(): returns a list of datum system objects"
    return getObjectsOfType("DatumSystem")

def getAllGeometricToleranceObjects():
    "getAllGeometricToleranceObjects(): returns a list of geometric tolerance objects"
    return getObjectsOfType("GeometricTolerance")

def getAllGDTObjects():
    "getAllGDTObjects(): returns a list of GDT objects"
    return getObjectsOfType(["AnnotationPlane","DatumFeature","DatumSystem","GeometricTolerance"])

def getAllAnnotationObjects():
    "getAllAnnotationObjects(): returns a list of annotation objects"
    return getObjectsOfType("Annotation")

def getParamType(param):
    if param in ["lineWidth"]:
        return "int"
    elif param in ["textFamily"]:
        return "string"
    elif param in ["textSize","lineScale"]:
        return "float"
    elif param in ["alwaysShowGrid","showUnit"]:
        return "bool"
    elif param in ["textColor","lineColor"]:
        return "unsigned"
    else:
        return None

def getParam(param,default=None):
    "getParam(parameterName): returns a GDT parameter value from the current config"
    p = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/GDT")
    t = getParamType(param)
    if t == "int":
        if default == None:
            default = 0
        return p.GetInt(param,default)
    elif t == "string":
        if default == None:
            default = ""
        return p.GetString(param,default)
    elif t == "float":
        if default == None:
            default = 1
        return p.GetFloat(param,default)
    elif t == "bool":
        if default == None:
            default = False
        return p.GetBool(param,default)
    elif t == "unsigned":
        if default == None:
            default = 0
        return p.GetUnsigned(param,default)
    else:
        return None

def setParam(param,value):
    "setParam(parameterName,value): sets a GDT parameter with the given value"
    p = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/GDT")
    t = getParamType(param)
    if t == "int": p.SetInt(param,value)
    elif t == "string": p.SetString(param,value)
    elif t == "float": p.SetFloat(param,value)
    elif t == "bool": p.SetBool(param,value)
    elif t == "unsigned": p.SetUnsigned(param,value)

#---------------------------------------------------------------------------
# General functions
#---------------------------------------------------------------------------

def stringencodecoin(ustr):
    """stringencodecoin(str): Encodes a unicode object to be used as a string in coin"""
    try:
        coin4 = coin.COIN_MAJOR_VERSION >= 4
    except (ImportError, AttributeError):
        coin4 = False
    if coin4:
        return ustr.encode('utf-8')
    else:
        return ustr.encode('latin1')

def stringplusminus():
    return ' ± ' if coin.COIN_MAJOR_VERSION >= 4 else ' +- '

def getType(obj):
    "getType(object): returns the GDT type of the given object"
    if not obj:
        return None
    if "Proxy" in obj.PropertiesList:
        if hasattr(obj.Proxy,"Type"):
            return obj.Proxy.Type
    return "Unknown"

def getObjectsOfType(typeList):
    "getObjectsOfType(string): returns a list of objects of the given type"
    listObjectsOfType = []
    objs = FreeCAD.ActiveDocument.Objects
    if not isinstance(typeList,list):
        typeList = [typeList]
    for obj in objs:
        for typ in typeList:
            if typ == getType(obj):
                listObjectsOfType.append(obj)
    return listObjectsOfType


def getRGB(param):
    color = QtGui.QColor(getParam(param,16753920)>>8)
    r = float(color.red()/255.0)
    g = float(color.green()/255.0)
    b = float(color.blue()/255.0)
    col = (r,g,b,0.0)
    return col

def getRGBText():
    return getRGB("textColor")

def getTextFamily():
    return getParam("textFamily","")

def getTextSize():
    return getParam("textSize",30)

def getLineWidth():
    return getParam("lineWidth",2)

def getRGBLine():
    return getRGB("lineColor")

def getPointsToPlot(obj):
    points = []
    segments = []
    if obj.GT != [] or obj.DF != None:
        X = FreeCAD.Vector(1.0,0.0,0.0)
        Y = FreeCAD.Vector(0.0,1.0,0.0)
        Direction = X if abs(X.dot(obj.AP.Direction)) < 0.8 else Y
        Vertical = obj.AP.Direction.cross(Direction).normalize()
        Horizontal = Vertical.cross(obj.AP.Direction).normalize()
        point = obj.selectedPoint
        d = point.distanceToPlane(obj.p1, obj.Direction)
        if obj.circumferenceBool:
            P3 = point + obj.Direction * (-d)
            d2 = (P3 - obj.p1) * Vertical
            P2 = obj.p1 + Vertical * (d2*3/4)
        else:
            P2 = obj.p1 + obj.Direction * (d*3/4)
            P3 = point
        points = [obj.p1, P2, P3]
        segments = [0,1,2]
        existGT = True
        if obj.GT != []:
            points, segments = getPointsToPlotGT(obj, points, segments, Vertical, Horizontal)
        else:
            existGT = False
        if obj.DF != None:
            points, segments = getPointsToPlotDF(obj, existGT, points, segments, Vertical, Horizontal)
        segments = segments + []
    return points, segments

def getPointsToPlotGT(obj, points, segments, Vertical, Horizontal):
    newPoints = points
    newSegments = segments
    if obj.ViewObject.LineScale > 0:
        sizeOfLine = obj.ViewObject.LineScale*7
    else:
        sizeOfLine = 1.0
    for i in range(len(obj.GT)):
        print('1029: ', obj.GT[i].DS.Primary)
        print('1030: ', obj.GT[i].DS.Secondary)
        print('1031: ', obj.GT[i].DS.Tertiary)
        d = len(newPoints)
        if points[2].x < points[0].x:
            P0 = newPoints[-1] + Vertical * (sizeOfLine) if i == 0 else FreeCAD.Vector(newPoints[-2])
        else:
            P0 = newPoints[-1] + Vertical * (sizeOfLine) if i == 0 else FreeCAD.Vector(newPoints[-1])
        P1 = P0 + Vertical * (-sizeOfLine*2)
        P2 = P0 + Horizontal * (sizeOfLine*2)
        P3 = P1 + Horizontal * (sizeOfLine*2)
        # print('901111', obj.GT[i].FeatureControlFrameIcon)

        lengthToleranceValue = len(stringencodecoin(displayExternal(obj.GT[i].ToleranceValue, obj.ViewObject.Decimals, 'Length', obj.ViewObject.ShowUnit)))
        if obj.GT[i].FeatureControlFrameIcon != '':
            lengthToleranceValue += 2
        if obj.GT[i].Circumference:
            lengthToleranceValue += 2
        P4 = P2 + Horizontal * (sizeOfLine*lengthToleranceValue)
        P5 = P3 + Horizontal * (sizeOfLine*lengthToleranceValue)
        if obj.GT[i].DS == None or obj.GT[i].DS.Primary == None:
            newPoints = newPoints + [P0, P2, P3, P4, P5, P1]
            newSegments = newSegments + [-1, 0+d, 3+d, 4+d, 5+d, 0+d, -1, 1+d, 2+d]
            if points[2].x < points[0].x:
                displacement = newPoints[-3].x - newPoints[-6].x
                for i in range(len(newPoints)-6, len(newPoints)):
                    newPoints[i].x-=displacement
        else:
            P6 = P4 + Horizontal * (sizeOfLine*2)
            P7 = P5 + Horizontal * (sizeOfLine*2)
            if obj.GT[i].DS.Secondary != None:
                P8 = P6 + Horizontal * (sizeOfLine*2)
                P9 = P7 + Horizontal * (sizeOfLine*2)
                if obj.GT[i].DS.Tertiary != None:
                    P10 = P8 + Horizontal * (sizeOfLine*2)
                    P11 = P9 + Horizontal * (sizeOfLine*2)
                    newPoints = newPoints + [P0, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P1]
                    newSegments = newSegments + [-1, 0+d, 9+d, 10+d, 11+d, 0+d, -1, 1+d, 2+d, -1, 3+d, 4+d, -1, 5+d, 6+d, -1, 7+d, 8+d]
                    if points[2].x < points[0].x:
                        displacement = newPoints[-3].x - newPoints[-12].x
                        for i in range(len(newPoints)-12, len(newPoints)):
                            newPoints[i].x-=displacement
                else:
                    newPoints = newPoints + [P0, P2, P3, P4, P5, P6, P7, P8, P9, P1]
                    newSegments = newSegments + [-1, 0+d, 7+d, 8+d, 9+d, 0+d, -1, 1+d, 2+d, -1, 3+d, 4+d, -1, 5+d, 6+d]
                    if points[2].x < points[0].x:
                        displacement = newPoints[-3].x - newPoints[-10].x
                        for i in range(len(newPoints)-10, len(newPoints)):
                            newPoints[i].x-=displacement
            else:
                newPoints = newPoints + [P0, P2, P3, P4, P5, P6, P7, P1]
                newSegments = newSegments + [-1, 0+d, 5+d, 6+d, 7+d, 0+d, -1, 1+d, 2+d, -1, 3+d, 4+d]
                if points[2].x < points[0].x:
                    displacement = newPoints[-3].x - newPoints[-8].x
                    for i in range(len(newPoints)-8, len(newPoints)):
                        newPoints[i].x-=displacement
    return newPoints, newSegments

def getPointsToPlotDF(obj, existGT, points, segments, Vertical, Horizontal):
    d = len(points)
    newPoints = points
    newSegments = segments
    if obj.ViewObject.LineScale > 0:
        sizeOfLine = obj.ViewObject.LineScale*7
    else:
        sizeOfLine = 1.0
    if not existGT:
        P0 = points[-1] + Vertical * (sizeOfLine)
        P1 = P0 + Horizontal * (sizeOfLine*2)
        P2 = P1 + Vertical * (-sizeOfLine*2)
        P3 = P2 + Horizontal * (-sizeOfLine*2)
        newPoints = newPoints + [P0, P1, P2, P3]
        newSegments = newSegments + [-1, 0+d, 1+d, 2+d, 3+d, 0+d]
        if points[2].x < points[0].x:
            displacement = newPoints[-2].x - newPoints[-1].x
            for i in range(len(newPoints)-4, len(newPoints)):
                newPoints[i].x-=displacement
    d=len(newPoints)
    P0 = newPoints[-1] + Horizontal * (sizeOfLine/2)
    P1 = P0 + Horizontal * (sizeOfLine)
    h = math.sqrt(sizeOfLine*sizeOfLine+(sizeOfLine/2)*(sizeOfLine/2))
    PAux = newPoints[-1] + Horizontal * (sizeOfLine)
    P2 = PAux + Vertical * (-h)
    P3 = PAux + Vertical * (-sizeOfLine*3)
    P4 = P3 + Horizontal * (sizeOfLine)
    P5 = P4 + Vertical * (-sizeOfLine*2)
    P6 = P5 + Horizontal * (-sizeOfLine*2)
    P7 = P6 + Vertical * (sizeOfLine*2)
    newPoints = newPoints + [P0, P1, P2, P3, P4, P5, P6, P7]
    newSegments = newSegments + [-1, 0+d, 2+d, -1, 1+d, 2+d, 3+d, 4+d, 5+d, 6+d, 7+d, 3+d]
    return newPoints, newSegments

def plotStrings(self, fp, points):
    if fp.ViewObject.LineScale > 0:
        sizeOfLine = fp.ViewObject.LineScale * 7
    else:
        sizeOfLine = 1.0
    X = FreeCAD.Vector(1.0,0.0,0.0)
    Y = FreeCAD.Vector(0.0,1.0,0.0)
    Direction = X if abs(X.dot(fp.AP.Direction)) < 0.8 else Y
    Vertical = fp.AP.Direction.cross(Direction).normalize()
    Horizontal = Vertical.cross(fp.AP.Direction).normalize()
    index = 0
    indexIcon = 0
    displacement = 0
    # print('938')
    if fp.GT != []:
        for i in range(len(fp.GT)):
            distance = 0
            # posToleranceValue
            v = (points[7+displacement] - points[5+displacement])
            if v.x != 0:
                distance = (v.x)/2
            elif v.y != 0:
                distance = (v.y)/2
            else:
                distance = (v.z)/2
            if fp.GT[i].FeatureControlFrameIcon != '':
                distance -= sizeOfLine
            if fp.GT[i].Circumference:
                distance += sizeOfLine
            centerPoint = points[5+displacement] + Horizontal * (distance)
            posToleranceValue = centerPoint + Vertical * (sizeOfLine/2)
            # posCharacteristic
            auxPoint = points[3+displacement] + Vertical * (-sizeOfLine*2)
            self.points[indexIcon].point.setValues([[auxPoint.x,auxPoint.y,auxPoint.z],[points[5+displacement].x,points[5+displacement].y,points[5+displacement].z],[points[4+displacement].x,points[4+displacement].y,points[4+displacement].z],[points[3+displacement].x,points[3+displacement].y,points[3+displacement].z]])
            self.face[indexIcon].numVertices = 4
            s = 1/(sizeOfLine*2)
            dS = FreeCAD.Vector(Horizontal) * s
            dT = FreeCAD.Vector(Vertical) * s
            self.svgPos[indexIcon].directionS.setValue(dS.x, dS.y, dS.z)
            self.svgPos[indexIcon].directionT.setValue(dT.x, dT.y, dT.z)
            displacementH = ((Horizontal*auxPoint)%(sizeOfLine*2))/(sizeOfLine*2)
            displacementV = ((Vertical*auxPoint)%(sizeOfLine*2))/(sizeOfLine*2)
            self.textureTransform[indexIcon].translation.setValue(-displacementH,-displacementV)
            filename = fp.GT[i].CharacteristicIcon
            filename = filename.replace(':/dd/icons', MBD_DIR)
            self.svg[indexIcon].filename = str(filename)
            indexIcon+=1
            # posFeactureControlFrame
            if fp.GT[i].FeatureControlFrameIcon != '':
                auxPoint1 = points[7+displacement] + Horizontal * (-sizeOfLine*2)
                auxPoint2 = auxPoint1 + Vertical * (sizeOfLine*2)
                self.points[indexIcon].point.setValues([[auxPoint1.x,auxPoint1.y,auxPoint1.z],[points[7+displacement].x,points[7+displacement].y,points[7+displacement].z],[points[6+displacement].x,points[6+displacement].y,points[6+displacement].z],[auxPoint2.x,auxPoint2.y,auxPoint2.z]])
                self.face[indexIcon].numVertices = 4
                self.svgPos[indexIcon].directionS.setValue(dS.x, dS.y, dS.z)
                self.svgPos[indexIcon].directionT.setValue(dT.x, dT.y, dT.z)
                displacementH = ((Horizontal*auxPoint1)%(sizeOfLine*2))/(sizeOfLine*2)
                displacementV = ((Vertical*auxPoint1)%(sizeOfLine*2))/(sizeOfLine*2)
                self.textureTransform[indexIcon].translation.setValue(-displacementH,-displacementV)
                filename = fp.GT[i].FeatureControlFrameIcon
                filename = filename.replace(':/dd/icons',MBD_DIR)
                self.svg[indexIcon].filename = str(filename)
                indexIcon+=1
                # print('988')
            # posDiameter
            if fp.GT[i].Circumference:
                auxPoint1 = points[5+displacement] + Horizontal * (sizeOfLine*2)
                auxPoint2 = auxPoint1 + Vertical * (sizeOfLine*2)
                self.points[indexIcon].point.setValues([[points[5+displacement].x,points[5+displacement].y,points[5+displacement].z],[auxPoint1.x,auxPoint1.y,auxPoint1.z],[auxPoint2.x,auxPoint2.y,auxPoint2.z],[points[4+displacement].x,points[4+displacement].y,points[4+displacement].z]])
                self.face[indexIcon].numVertices = 4
                self.svgPos[indexIcon].directionS.setValue(dS.x, dS.y, dS.z)
                self.svgPos[indexIcon].directionT.setValue(dT.x, dT.y, dT.z)
                displacementH = ((Horizontal*points[5+displacement])%(sizeOfLine*2))/(sizeOfLine*2)
                displacementV = ((Vertical*points[5+displacement])%(sizeOfLine*2))/(sizeOfLine*2)
                self.textureTransform[indexIcon].translation.setValue(-displacementH,-displacementV)
                filename = os.path.join(MBD_DIR, 'icons', 'diameter.svg')
                self.svg[indexIcon].filename = str(filename)
                indexIcon+=1

            self.textGT[index].string = self.textGT3d[index].string = stringencodecoin(displayExternal(fp.GT[i].ToleranceValue, fp.ViewObject.Decimals, 'Length', fp.ViewObject.ShowUnit))
            self.textGTpos[index].translation.setValue([posToleranceValue.x, posToleranceValue.y, posToleranceValue.z])
            self.textGT[index].justification = coin.SoAsciiText.CENTER
            index+=1
            displacement+=6
            if fp.GT[i].DS != None and fp.GT[i].DS.Primary != None:
                if fp.GT[i].FeatureControlFrameIcon != '':
                    distance += (sizeOfLine*2)
                if fp.GT[i].Circumference:
                    distance -= (sizeOfLine*2)
                posPrimary = posToleranceValue + Horizontal * (distance+sizeOfLine)
                self.textGT[index].string = self.textGT3d[index].string = str(fp.GT[i].DS.Primary.Label)
                self.textGTpos[index].translation.setValue([posPrimary.x, posPrimary.y, posPrimary.z])
                self.textGT[index].justification = coin.SoAsciiText.CENTER
                index+=1
                displacement+=2
                if fp.GT[i].DS.Secondary != None:
                    posSecondary = posPrimary + Horizontal * (sizeOfLine*2)
                    self.textGT[index].string = self.textGT3d[index].string = str(fp.GT[i].DS.Secondary.Label)
                    self.textGTpos[index].translation.setValue([posSecondary.x, posSecondary.y, posSecondary.z])
                    self.textGT[index].justification = coin.SoAsciiText.CENTER
                    index+=1
                    displacement+=2
                    if fp.GT[i].DS.Tertiary != None:
                        posTertiary = posSecondary + Horizontal * (sizeOfLine*2)
                        self.textGT[index].string = self.textGT3d[index].string = str(fp.GT[i].DS.Tertiary.Label)
                        self.textGTpos[index].translation.setValue([posTertiary.x, posTertiary.y, posTertiary.z])
                        self.textGT[index].justification = coin.SoAsciiText.CENTER
                        index+=1
                        displacement+=2
        if fp.circumferenceBool and True in [l.Circumference for l in fp.GT]:
            # posDiameterTolerance
            auxPoint1 = FreeCAD.Vector(points[4])
            auxPoint2 = auxPoint1 + Horizontal * (sizeOfLine*2)
            auxPoint3 = auxPoint2 + Vertical * (sizeOfLine*2)
            auxPoint4 = auxPoint1 + Vertical * (sizeOfLine*2)
            self.points[indexIcon].point.setValues([[auxPoint1.x,auxPoint1.y,auxPoint1.z],[auxPoint2.x,auxPoint2.y,auxPoint2.z],[auxPoint3.x,auxPoint3.y,auxPoint3.z],[auxPoint4.x,auxPoint4.y,auxPoint4.z]])
            self.face[indexIcon].numVertices = 4
            self.svgPos[indexIcon].directionS.setValue(dS.x, dS.y, dS.z)
            self.svgPos[indexIcon].directionT.setValue(dT.x, dT.y, dT.z)
            displacementH = ((Horizontal*auxPoint1)%(sizeOfLine*2))/(sizeOfLine*2)
            displacementV = ((Vertical*auxPoint1)%(sizeOfLine*2))/(sizeOfLine*2)
            self.textureTransform[indexIcon].translation.setValue(-displacementH,-displacementV)
            filename = os.path.join(MBD_DIR, 'icons', 'diameter.svg')
            self.svg[indexIcon].filename = str(filename)
            indexIcon+=1
            posDiameterTolerance = auxPoint2 + Vertical * (sizeOfLine/2)
            self.textGT[index].justification = coin.SoAsciiText.LEFT
            self.textGTpos[index].translation.setValue([posDiameterTolerance.x, posDiameterTolerance.y, posDiameterTolerance.z])
            text = stringencodecoin(displayExternal(fp.diameter, fp.ViewObject.Decimals, 'Length', fp.ViewObject.ShowUnit) + stringplusminus() + displayExternal(fp.toleranceDiameter, fp.ViewObject.Decimals, 'Length', fp.ViewObject.ShowUnit))
            self.textGT[index].string = self.textGT3d[index].string = text
            index+=1
        for i in range(index):
            try:
                DirectionAux = FreeCAD.Vector(fp.AP.Direction)
                DirectionAux.x = abs(DirectionAux.x)
                DirectionAux.y = abs(DirectionAux.y)
                DirectionAux.z = abs(DirectionAux.z)
                rotation=(DraftGeomUtils.getRotation(DirectionAux)).Q
                self.textGTpos[i].rotation.setValue(rotation)
            except:
                pass
        for i in range(index,len(self.textGT)):
            if str(self.textGT[i].string) != "":
                self.textGT[i].string = self.textGT3d[i].string = ""
            else:
                break
        for i in range(indexIcon,len(self.svg)):
            if str(self.face[i].numVertices) != 0:
                self.face[i].numVertices = 0
                self.svg[i].filename = ""
    else:
        for i in range(len(self.textGT)):
            if str(self.textGT[i].string) != "" or str(self.svg[i].filename) != "":
                self.textGT[i].string = self.textGT3d[i].string = ""
                self.face[i].numVertices = 0
                self.svg[i].filename = ""
                # print('1081')
            else:
                break
    if fp.DF != None:
        # print(fp.DF, '1085')
        self.textDF.string = self.textDF3d.string = str(fp.DF.Label)
        distance = 0
        v = (points[-3] - points[-2])
        if v.x != 0:
            distance = (v.x)/2
        elif v.y != 0:
            distance = (v.y)/2
        else:
            distance = (v.z)/2
        centerPoint = points[-2] + Horizontal * (distance)
        centerPoint = centerPoint + Vertical * (sizeOfLine/2)
        self.textDFpos.translation.setValue([centerPoint.x, centerPoint.y, centerPoint.z])
        # print('1098')
        try:
            DirectionAux = FreeCAD.Vector(fp.AP.Direction)
            DirectionAux.x = abs(DirectionAux.x)
            DirectionAux.y = abs(DirectionAux.y)
            DirectionAux.z = abs(DirectionAux.z)
            rotation=(DraftGeomUtils.getRotation(DirectionAux)).Q
            self.textDFpos.rotation.setValue(rotation)
        except:
            # print('1106')
            pass
    else:
        self.textDF.string = self.textDF3d.string = ""
    # print('WHY DOES IT STOP WORKING HERE')
    # print(fp.DF, '1110')
    if fp.GT != [] or fp.DF != None:
        # print('1111')
        if len(fp.faces) > 1:
            # posNumFaces
            # print('1113')
            centerPoint = points[3] + Horizontal * (sizeOfLine)
            posNumFaces = centerPoint + Vertical * (sizeOfLine/2)
            self.textGT[index].string = self.textGT3d[index].string = (str(len(fp.faces))+'x')
            self.textGTpos[index].translation.setValue([posNumFaces.x, posNumFaces.y, posNumFaces.z])
            self.textGT[index].justification = coin.SoAsciiText.CENTER
            try:
                DirectionAux = FreeCAD.Vector(fp.AP.Direction)
                DirectionAux.x = abs(DirectionAux.x)
                DirectionAux.y = abs(DirectionAux.y)
                DirectionAux.z = abs(DirectionAux.z)
                rotation=(DraftGeomUtils.getRotation(DirectionAux)).Q
                self.textGTpos[index].rotation.setValue(rotation)
            except:
                # print('1127')
                pass
            index+=1
def load_json_file(file_path):
    try:
        # First attempt: Try to load the file using the given path
        with open(file_path, 'r') as file:
            freecad_data = json.load(file)
        print("- File loaded successfully using the original path.")
        return freecad_data
    except FileNotFoundError:
        print("- Path in Windows format does not work, trying path in Linux format now.")
        # Modify the path for Linux (WSL)
        modified_path = file_path.replace('\\', '/')  # Change backslashes to forward slashes
        if modified_path.startswith('C:'):
            modified_path = '/mnt/c' + modified_path[2:]  # Replace 'C:' with '/mnt/c'
        try:
            # Second attempt: Try to load the file using the modified path
            with open(modified_path, 'r') as file:
                freecad_data = json.load(file)
            print("- File loaded successfully using the modified Linux path.")
            return freecad_data
        except FileNotFoundError:
            print("- File could not be loaded using either Windows or Linux paths.")
        except Exception as e:
            # Catch other possible exceptions and print the error
            print(f"- An error occurred: {e}")
    except Exception as e:
        # Catch other possible exceptions and print the error during the first attempt
        print(f"- An error occurred during the first attempt: {e}")
def displayExternal(internValue,decimals=4,dim='Length',showUnit=True):
    '''return an internal value (ie mm) Length or Angle converted for display according
    to Units Schema in use.'''

    if dim == 'Length':
        qty = FreeCAD.Units.Quantity(internValue,FreeCAD.Units.Length)
        pref = qty.getUserPreferred()
        # conversion = pref[1]
        conversion = 1
        uom = 'mm'
        # uom = pref[2]
    elif dim == 'Angle':
        qty = FreeCAD.Units.Quantity(internValue,FreeCAD.Units.Angle)
        pref=qty.getUserPreferred()
        # conversion = pref[1]
        conversion = 1
        uom = 'mm'
    else:
        conversion = 1.0
        uom = "??"
    if not showUnit:
        uom = ""
    print(internValue, conversion, uom, '1368')
    fmt = "{0:."+ str(decimals) + "f} "+ uom
    print(fmt, '1370')
    displayExt = fmt.format(float(internValue) / float(conversion))
    print(displayExt, '1372')
    displayExt = displayExt.replace(".",QtCore.QLocale().decimalPoint())
    print(displayExt, '1373')
    return displayExt
import shutil
def take_screenshot(filename, output_dir, width=1500, height=1500):
    '''
    Takes screenshot from different camera views and saves them in the output directory
    '''
    step_name = filename.split('\\')
    if len(step_name) == 1:
        step_name = filename.split('/')
    step_name = step_name[-1]
    print('1399:', output_dir, step_name)
    if os.path.exists(os.path.join(output_dir, step_name)):
        shutil.rmtree(os.path.join(output_dir, step_name))

    if os.path.exists(os.path.join(output_dir, step_name)) == False:
        os.makedirs(os.path.join(output_dir, step_name))


    output_dir = os.path.join(output_dir, step_name)
    
    print('1409: ', output_dir)

    for p in ['PerspectiveCamera','OrthographicCamera']:
        Gui.SendMsgToActiveView(p)
        for f in ['ViewAxo','ViewFront','ViewTop']:
            Gui.SendMsgToActiveView(f)
            for x,y in [[width,height]]:
                transparent_file_path = os.path.join(output_dir,f'{step_name}_{p}_{f}_{x}_{y}.png').replace('\\', '/')
                # print('1375: ', white_file_path)
                # transparent_file_path = f'{output_dir}/{step_name}_{p}_{f}_{x}_{y}.png'
                print(transparent_file_path)
                # Gui.ActiveDocument.ActiveView.saveImage(white_file_path,x,y,'White')
                Gui.ActiveDocument.ActiveView.saveImage(transparent_file_path,x,y,'Transparent')
                
        print(f'{p} done!')
        
import threading
def stop_gui_loop():
    Gui.closeApplication()
def main(filename, datum_path, gdt_path, screenshot_dir):
    
    # filename = "C:\\Users\\Drishti Bhasin\\Downloads\\GD&T Example 02.stp"
    model = load_step_file(filename)
# List to hold faces that are considered as part of the top view
    top_view_faces = []
    # print('hellooooo')
    # Iterate through all objects in the imported document
    # Iterate through all objects in the imported document
    for obj in FreeCAD.ActiveDocument.Objects:
        # Check if the object has a Shape attribute
        if hasattr(obj, "Shape"):
            # Iterate through all faces of the object
            for i, face in enumerate(obj.Shape.Faces):
                # Calculate the normal vector at the center of the face
                u, v = face.ParameterRange[1:], face.ParameterRange[3:]
                u = sum(u) / 2.0
                v = sum(v) / 2.0
                normal = face.normalAt(u, v)
                # Check if the Z-component of the normal vector is positive
                if normal.z > 0:
                    # Append a tuple of the face and its name (e.g., 'Face1', 'Face2', etc.)
                    face_name = 'Face' + str(i + 1)  # FreeCAD faces are 1-indexed in the GUI
                    top_view_faces.append((face, face_name))

    # Now, top_view_faces contains all faces that are oriented towards the top view
    print(f"Found {len(top_view_faces)} faces belonging to the top view.")
    face_data = extract_face_data()
    # print(face_data)
# Calculate the center of mass for the part
    center_of_mass = model.Shape.CenterOfMass
    
    # Function to find a top face normal vector
    def find_top_face_normal(objs):
        for obj in objs:
            if hasattr(obj, "Shape"):
                for face in obj.Shape.Faces:
                    u, v = face.ParameterRange[1:], face.ParameterRange[3:]
                    u = sum(u) / 2.0
                    v = sum(v) / 2.0
                    normal = face.normalAt(u, v)
                    if normal.z > 0:
                        return normal
        return None

    normal_vector = find_top_face_normal(FreeCAD.ActiveDocument.Objects)

    print(datum_path, '1425')
    datums = load_json_file(datum_path)
    gdt_info = load_json_file(gdt_path)
    df_labels = ['D', 'E', 'F']
    datum_feats = datums.keys()
    datum_mapping = {}
    inv_datum_mapping = {}
    dfs = {}
    if datum_feats:
        for i, face in enumerate(datum_feats):
            # print(datums[face])
            obj = makeDatumFeature(df_labels[i])
            dfs[face] = obj
            datum_mapping[face] = datums[face][0]
            inv_datum_mapping[datums[face][0]] = face
    print('1470: ', datum_feats)
    print('1471: ', datum_mapping)
    print('1472: ', dfs)
    # print('1406', dfs)
    # print('1407', datum_mapping)
    print('1408', gdt_info)
    geo_tol = {}
    geo_tol_keys = ['faces',
        'Characteristic',
        'Circumference',
        'Tolerance_value',
        'ds',
        'df',
        'Feature-Control-Frame',
        'object']
    for i, face in enumerate(gdt_info):
        label = 'GT'+str(i+1)
        geo_tol[label] = {}
        gdt_face = gdt_info[face]
        for key in geo_tol_keys:
            if key == 'faces':
                geo_tol[label][key] = face
            elif key == 'ds':
                ds_label = ''
                ds_obj = {}
                gdt_ds = gdt_face['ds']
                if len(gdt_ds) > 0:
                    for df in gdt_ds:
                        ds_label += df
                        ds_obj[datum_mapping[df]] = dfs[df]
                geo_tol[label][key] = makeDatumSystem(ds_obj, ds_label)  
            elif key == 'df':
                if face in dfs:
                    geo_tol[label][key] = dfs[face]
                else:
                    geo_tol[label][key] = None 
                    
            else:
                geo_tol[label][key] = gdt_face.get(key, None)
        
    print('1437', geo_tol)
        
    # dictionary
    model_object = {
    'annotation_frame': {'AP1':{
        'annotation_plane': {
            'p1': {'x': center_of_mass.x, 'y': center_of_mass.y, 'z': center_of_mass.z},
            'faces': top_view_faces[0][1],
            'offset': 0,
            'direction':{'x': normal_vector.x, 'y': normal_vector.y, 'z': normal_vector.z},
        }, 'object': None},
    },
    'geometric_tolerance': geo_tol
    }
    
    # make annotation plane
    model_object['annotation_frame']['AP1']['object'] = makeAnnotationPlane(model_object, model, face_data, 'AP1')

    for label in geo_tol:
        gt = makeGeometricTolerance(model_object, model, label, DF=geo_tol[label]['df'], AP='AP1') 


    makeDimension(model)
    print('1553 is being called')
    # take_screenshot(filename, screenshot_dir)
    # print('1398 called her dimensions')
    # timer = threading.Timer(10.0, stop_gui_loop)

    # # Start the timer
    # timer.start()

    # # Start the GUI loop
    # Gui.exec_loop()

    # # After exiting the loop, cancel the timer in case it's still running
    # timer.cancel()

    # make the annotation object 
    
    # create object


if __name__ == "__main__":
    print('1552 started')
    from config import current_dir_linux, current_dir_win
    filename = sys.argv[1]
    datums = os.path.join(current_dir_win, "outputs/datum_analysis.json")
    gdt_info = os.path.join(current_dir_win, "outputs/ref_datum_analysis.json")
    screenshot_dir = os.path.join(current_dir_win, "mvp_result")
    # filename = sys.argv[1]
    # datums = sys.argv[2]
    # gdt_info = sys.argv[3]
    # screenshot_dir = sys.argv[4]
    main(filename, datums, gdt_info, screenshot_dir)


