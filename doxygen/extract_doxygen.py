import os
import networkx as nx
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_function_code(file_path, bodystart, bodyend):
    try:
        # 打开文件并读取所有行
        with open(file_path, 'r', errors='ignore') as file:
            lines = file.readlines()

        # 提取指定行号之间的代码（注意：bodystart 和 bodyend 是从1开始计数的）
        function_code = lines[bodystart - 1 : bodyend]

        # 将提取到的代码行合并成一个字符串
        return ''.join(function_code)
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

class SubGraph:
    """
    一个子图对应一个.dot文件，都是处理好的函数调用关系图
    """
    def __init__(self):
        self.index_to_funcname = {}
        self.index_to_code = {}
        self.funcname_to_index = {}
        self.funcname_to_code = {}
        self.adj = []
    
    def add_node(self, funcname, code):
        if funcname in self.funcname_to_index:
            # print(f"Function {funcname} already exists.")
            return
        # 获取当前节点数量，用于分配新的索引
        index = len(self.index_to_funcname)
        # 映射函数名和索引
        self.index_to_funcname[index] = funcname
        self.index_to_code[index] = code
        self.funcname_to_index[funcname] = index
        self.funcname_to_code[funcname] = code
        # 动态扩展邻接矩阵
        for row in self.adj:
            row.append(0)  # 给现有的每一行添加一个 0
        self.adj.append([0] * (index + 1))  # 添加新的行，对应新的函数节点
    
    def add_edge(self, from_func, to_func):
        if from_func not in self.funcname_to_index or to_func not in self.funcname_to_index:
            # print("One or both functions not found.")
            return
        
        from_index = self.funcname_to_index[from_func]
        to_index = self.funcname_to_index[to_func]
        
        # 更新邻接矩阵，表示函数调用关系
        self.adj[from_index][to_index] = 1
    
    def get_all_functions_name(self):
        func_names = list()
        for i in range(len(self.index_to_funcname)):
            func_names.append(self.index_to_funcname[i])
        return func_names

    def get_all_functions_code(self):
        func_codes = list()
        for i in range(len(self.index_to_code)):
            func_codes.append(self.index_to_code[i])
        return func_codes

    def to_dict(self):
        """将 SubGraph 转换为字典以便序列化为 JSON。"""
        data = vars(self)  # 获取对象的属性字典
        return data
    
    def find_top_nodes(self):
        num_nodes = len(self.index_to_funcname)
        in_degrees = [0] * num_nodes
        out_degrees = [0] * num_nodes
        # 计算入度和出度
        for i in range(num_nodes):
            for j in range(num_nodes):
                if self.adj[i][j] != 0:
                    # i 调用了 j
                    out_degrees[i] += 1
                    in_degrees[j] += 1
        # 找到入度为 0 且出度大于 0 的节点
        top_level_nodes = []
        for i in range(num_nodes):
            if in_degrees[i] == 0 and out_degrees[i] > 0:
                func_name = self.index_to_funcname[str(i)]  # 注意索引是字符串
                top_level_nodes.append(func_name)
        return top_level_nodes

    def find_callees(self, func_name):
        if func_name not in self.funcname_to_index:
            raise ValueError(f"Function {func_name} not found.")
        index = self.funcname_to_index[func_name]
        callees = []
        for callee_index, is_called in enumerate(self.adj[index]):
            if is_called:
                callee_name = self.index_to_funcname[str(callee_index)]
                callees.append(callee_name)
        return callees
    
    def dfs(self, current_node, visited=None):
        if visited is None:
            visited = set()
        for neighbor in range(len(self.adj[current_node])):
            if self.adj[current_node][neighbor] and neighbor not in visited:
                visited.add(neighbor)
                self.dfs(neighbor, visited)
        return visited
    
class Graph:
    """
    总图是一个项目的所有子图的集合
    """
    def __init__(self, project_path) -> None:
        self.project_path = project_path
        self.project_name = project_path.split('/')[-1]
        self.data = []

    def build_data(self):
        # 检查目录下是否存在 Doxygen 生成的文件
        if not os.path.exists(os.path.join(self.project_path, "Doxyfile")):
            raise FileNotFoundError("Doxygen 生成的文件不存在")
        # 获取project_path/doxygen/html目录下的所有cgraph.dot文件
        cgraphs = [f for f in os.listdir(os.path.join(self.project_path, "doxygen", "html")) if f.endswith("_cgraph.dot") and not f.startswith("class") and not f.startswith("struct")]
        if cgraphs:
            # 解析cgraph.dot文件
            self.nodes = {}
            for cg in tqdm(cgraphs):
                try:
                    self.deal_dot(cg)
                except:
                    continue
                

    def deal_dot(self, cg):
        subgraph = SubGraph()
        # 处理cgraph.dot文件
        if cg.endswith('_cgraph.dot'):    
            G = nx.nx_agraph.read_dot(os.path.join(self.project_path, "doxygen", "html", cg))
            # NODE1代表主函数
            main_func_name = G.nodes(data=True)['Node1']['label']
            cgraph_batch_nodes_key_map = {}
            # print(f"Processing {cg}...")
            for node in G.nodes(data=True):
                node_name = node[0]
                node_key = self.find_code_by_dot_node(node, cg, subgraph)
                cgraph_batch_nodes_key_map[node_name] = node_key
            # 添加边
            for edge in G.edges():
                source = cgraph_batch_nodes_key_map[edge[0]]
                target = cgraph_batch_nodes_key_map[edge[1]]
                subgraph.add_edge(source, target)
        
            # 找到icgraph.dot文件
            icg = cg.replace('_cgraph', '_icgraph')
            # print(f"Processing {icg}...")
            icgraph_batch_nodes_key_map = {}
            if os.path.exists(os.path.join(self.project_path, "doxygen", "html", icg)):
                G = nx.nx_agraph.read_dot(os.path.join(self.project_path, "doxygen", "html", icg))
                for node in G.nodes(data=True):
                    node_name = node[0]
                    node_key = self.find_code_by_dot_node(node, icg, subgraph)
                    icgraph_batch_nodes_key_map[node_name] = node_key
                for edge in G.edges():
                    source = icgraph_batch_nodes_key_map[edge[1]]
                    target = icgraph_batch_nodes_key_map[edge[0]]
                    subgraph.add_edge(source, target)
        elif cg.endswith('_icgraph.dot'):
            icgraph_batch_nodes_key_map = {}
            if os.path.exists(os.path.join(self.project_path, "doxygen", "html", cg)):
                G = nx.nx_agraph.read_dot(os.path.join(self.project_path, "doxygen", "html", cg))
                for node in G.nodes(data=True):
                    node_name = node[0]
                    node_key = self.find_code_by_dot_node(node, cg, subgraph)
                    icgraph_batch_nodes_key_map[node_name] = node_key
                for edge in G.edges():
                    source = icgraph_batch_nodes_key_map[edge[1]]
                    target = icgraph_batch_nodes_key_map[edge[0]]
                    subgraph.add_edge(source, target)
        self.data.append(subgraph)

    def find_code_by_dot_node(self, node, cg, subgraph):
        """
        返回一个函数的key值, 为函数名
        """
        import xml.etree.ElementTree as ET
        if "URL" in node[1].keys():
            func_name = node[1]['label']
            func_file_name = node[1]['URL'].split('.html')[0].split('$')[-1]
            func_hash = node[1]['URL'].split('#')[-1]
        else:
            func_name = node[1]['label']
            func_file_name = cg.rsplit('_', 2)[0]
            func_hash = cg.rsplit('_', 2)[1]
        
        # 到project_path/doxygen/xml目录下找到对应的xml文件
        xml_file = os.path.join(self.project_path, "doxygen", "xml", f"{func_file_name}.xml")
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for memberdef in root.findall(".//memberdef[@kind='function']"):
            id_attr = memberdef.attrib.get('id')
            if id_attr == f"{func_file_name}_1{func_hash}":
                location = memberdef.find('location')
                # 看一下有没有bodyfile
                if location.attrib.get('bodyfile') == None:
                    return func_name
                body_start = int(location.attrib.get('bodystart'))  # 提取 bodystart 属性
                body_end = int(location.attrib.get('bodyend'))      # 提取 bodyend 属性
                file = os.path.join(self.project_path, location.attrib.get('bodyfile'))
                code = extract_function_code(file, body_start, body_end)
                subgraph.add_node(func_name, code)
                return func_name

    
    def to_json(self, cache_dir):
        """将 Graph 中的所有 SubGraph 序列化为 JSON。"""
        subgraphs_dict = [subgraph.to_dict() for subgraph in self.data]
        with open(f"{cache_dir}", "w") as f:
            json.dump(subgraphs_dict, f, indent=4)

def has_xml_files(directory):
    try:
        # 列出目录中的所有文件
        files = os.listdir(directory)
        # 检查是否有以 .xml 结尾的文件
        return any(file.endswith('.xml') for file in files)
    except FileNotFoundError:
        print(f"目录 {directory} 不存在")
        return False

def build_data():
    project_root = "" # your project root path
    projects = os.listdir(project_root)
    
    # 创建线程池执行器
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for project in projects:
            futures.append(executor.submit(process_project, project, project_root))
        
        # 处理完成的任务
        for future in as_completed(futures):
            project_result = future.result()
            print(f"Finished processing project: {project_result}")

def process_project(project, project_root):
    print(f"Processing project: {project}")
    if os.path.exists(f"cache/{project}-subgraphs.json"):
        print(f"Project {project} already processed.")
        return project
    if not has_xml_files(os.path.join(project_root, project, "doxygen", "xml")):
        print(f"Project not processed.")
        return project
    graph = Graph(os.path.join(project_root, project))
    graph.build_data()
    return project

if __name__ == "__main__":
    build_data()