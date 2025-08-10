import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem):
    """将XML元素格式化为可读字符串"""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def extract_arm_structure(arm_xml, arm_side):
    """从手臂XML中提取链接和关节结构"""
    tree = ET.parse(arm_xml)
    root = tree.getroot()
    body = root.find('.//body')
    
    links = []
    joints = []
    
    # 递归提取身体元素
    def process_body(body_elem, parent_link=None):
        # 创建链接
        link_name = body_elem.get('name') + '_' + arm_side
        link = ET.Element('link', name=link_name)
        
        # 提取惯性数据
        inertial = body_elem.find('inertial')
        if inertial is not None:
            inertial_elem = ET.SubElement(link, 'inertial')
            origin = inertial.find('origin')
            if origin is not None:
                ET.SubElement(inertial_elem, 'origin', 
                             xyz=origin.get('pos') if 'pos' in origin.attrib else "0 0 0",
                             rpy=origin.get('quat') if 'quat' in origin.attrib else "0 0 0")
            mass = inertial.find('mass')
            if mass is not None:
                ET.SubElement(inertial_elem, 'mass', value=mass.get('mass'))
            inertia = inertial.find('inertia')
            if inertia is not None:
                ET.SubElement(inertial_elem, 'inertia',
                             ixx=inertia.get('diaginertia').split()[0],
                             ixy="0", ixz="0",
                             iyy=inertia.get('diaginertia').split()[1],
                             iyz="0",
                             izz=inertia.get('diaginertia').split()[2])
        
        # 提取几何网格 mesh
        for geom in body_elem.findall('geom'):
            mesh = geom.get('mesh')
            if mesh:
                visual = ET.SubElement(link, 'visual')
                origin = ET.SubElement(visual, 'origin', 
                                      xyz=geom.get('pos') if 'pos' in geom.attrib else "0 0 0",
                                      rpy=geom.get('euler') if 'euler' in geom.attrib else "0 0 0")
                geometry = ET.SubElement(visual, 'geometry')
                ET.SubElement(geometry, 'mesh', filename=f"package://meshes/{mesh}.STL")
        
        links.append(link)
        
        # 处理关节
        joint = body_elem.find('joint')
        if joint is not None and parent_link is not None:
            joint_name = joint.get('name') + '_' + arm_side
            joint_elem = ET.Element('joint', name=joint_name, type=joint.get('type', 'fixed'))
            
            # 关节限制
            if 'range' in joint.attrib:
                limits = joint.get('range').split()
                ET.SubElement(joint_elem, 'limit', 
                             lower=limits[0], upper=limits[1], 
                             effort="100", velocity="1")
            
            # 父子链接关系
            ET.SubElement(joint_elem, 'parent', link=parent_link.get('name'))
            ET.SubElement(joint_elem, 'child', link=link.get('name'))
            
            # 关节轴
            axis = joint.get('axis')
            if axis:
                ET.SubElement(joint_elem, 'axis', xyz=axis)
            
            joints.append(joint_elem)
        
        # 递归处理子身体
        for child_body in body_elem.findall('body'):
            process_body(child_body, link)
    
    process_body(body)
    return links, joints

def main():
    # 创建URDF root
    robot = ET.Element('robot', name="bimanual_piper")
    
    # 添加底座链接
    base_link = ET.Element('link', name="base_link")
    visual = ET.SubElement(base_link, 'visual')
    geometry = ET.SubElement(visual, 'geometry')
    ET.SubElement(geometry, 'mesh', filename="package://meshes/base_link_aloha.STL")
    robot.append(base_link)
    
    # 添加桌子链接
    table_link = ET.Element('link', name="table")
    visual = ET.SubElement(table_link, 'visual')
    geometry = ET.SubElement(visual, 'geometry')
    ET.SubElement(geometry, 'box', size="0.485 0.295 0.47")
    robot.append(table_link)
    
    # 桌子固定关节
    table_joint = ET.Element('joint', name="table_joint", type="fixed")
    ET.SubElement(table_joint, 'parent', link="base_link")
    ET.SubElement(table_joint, 'child', link="table")
    ET.SubElement(table_joint, 'origin', xyz="0.7 0.0 0.4", rpy="0 0 1.57")
    robot.append(table_joint)
    
    # 处理左臂
    left_links, left_joints = extract_arm_structure('./piper_robot_xml/piper_left.xml', 'left')
    for link in left_links:
        robot.append(link)
    for joint in left_joints:
        robot.append(joint)
    
    # 左臂基座关节
    left_base_joint = ET.Element('joint', name="left_base_joint", type="fixed")
    ET.SubElement(left_base_joint, 'parent', link="base_link")
    ET.SubElement(left_base_joint, 'child', link="piper_left_left")
    ET.SubElement(left_base_joint, 'origin', xyz="0.234 0.303 0.761", rpy="0 0 0")
    robot.append(left_base_joint)
    
    # 处理右臂
    right_links, right_joints = extract_arm_structure('./piper_robot_xml/piper_right.xml', 'right')
    for link in right_links:
        robot.append(link)
    for joint in right_joints:
        robot.append(joint)
    
    # 右臂基座关节
    right_base_joint = ET.Element('joint', name="right_base_joint", type="fixed")
    ET.SubElement(right_base_joint, 'parent', link="base_link")
    ET.SubElement(right_base_joint, 'child', link="piper_right_right")
    ET.SubElement(right_base_joint, 'origin', xyz="0.234 -0.303 0.761", rpy="0 0 0")
    robot.append(right_base_joint)
    
    # 保存URDF文件
    with open('bimanual_piper.urdf', 'w') as f:
        f.write(prettify(robot))

if __name__ == "__main__":
    main()