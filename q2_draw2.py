from graphviz import Digraph

def create_regular_pso_flowchart():
    """
    在用户认可的优雅 Graphviz 风格基础上，
    通过微调使布局更加规整、稳定。
    """
    dot = Digraph(comment='PSO Algorithm Flowchart - Regular Layout')
    
    # --- 基础设置 (与您提供的版本一致) ---
    dot.attr('graph', 
             rankdir='TB', 
             bgcolor='transparent',
             fontname='Microsoft YaHei', # 确保中文字体
             splines='spline',
             nodesep='0.8', # 稍微增加垂直间距
             ranksep='1.2') # 稍微增加水平间距

    dot.attr('node', 
             shape='box', 
             style='filled, rounded', 
             fontname='Microsoft YaHei', 
             fontsize='14',
             color='#37474F',
             fontcolor='#263238')
    
    dot.attr('edge', 
             fontname='Microsoft YaHei', 
             fontsize='12',
             color='#546E7A')

    # --- 节点定义 ---
    dot.node('start', '开始', shape='ellipse', fillcolor='#E0E0E0')
    dot.node('end', '输出最优解\n(全局最优解 gbest)', shape='ellipse', fillcolor='#FFCDD2')

    # 阶段一：初始化 (使用子图进行视觉分组)
    with dot.subgraph(name='cluster_init') as c:
        c.attr(label='第一阶段：初始化', style='filled', color='#ECEFF1', fontname='Microsoft YaHei', fontsize='16')
        
        # --- 关键修正 1：将项目符号 '•' 替换为 '-' ---
        c.node('init_problem', '定义优化问题\n(目标函数与约束条件)', shape='parallelogram', fillcolor='#CFD8DC')
        c.node('init_swarm', '初始化粒子群\n- 随机生成粒子位置\n- 随机生成粒子速度', fillcolor='#CFD8DC')
        c.node('init_eval', '评估初始适应度\n(并行计算各粒子目标值)', fillcolor='#CFD8DC')
        c.node('init_best', '设置初始最优值\n- 个体最优 = 当前位置\n- 全局最优 = 群体最佳', fillcolor='#CFD8DC')

    # 阶段二：迭代优化
    with dot.subgraph(name='cluster_loop') as c:
        c.attr(label='第二阶段：迭代优化', style='filled', color='#FFF8E1', fontname='Microsoft YaHei', fontsize='16')
        c.node('update_velocity', 
               label='<速度更新<br/>v[t+1] = ωv[t] + c₁r₁(pbest - x) + c₂r₂(gbest - x)>',
               shape='box', fillcolor='#B0BEC5')
        c.node('update_position', 
               label='<位置更新<br/>x[t+1] = x[t] + v[t+1] (应用边界约束)>',
               shape='box', fillcolor='#B0BEC5')
        c.node('eval_fitness', '调用仿真模型\n计算新位置的适应度值', shape='cylinder', fillcolor='#B2DFDB')
        c.node('update_pbest', '更新个体最优\n(pbest)', shape='diamond', style='filled', fillcolor='#BBDEFB')
        c.node('update_gbest', '更新全局最优\n(gbest)', shape='diamond', style='filled', fillcolor='#BBDEFB')

    # --- 使用 rank='same' 强制节点水平对齐 (核心布局方法) ---
    dot.body.append("{ rank = same; init_problem; update_velocity }")
    dot.body.append("{ rank = same; init_swarm; update_position }")
    dot.body.append("{ rank = same; init_eval; eval_fitness }")
    dot.body.append("{ rank = same; init_best; update_pbest }")
    dot.body.append("{ rank = same; end; update_gbest }") # 让 gbest 和 end 对齐

    # --- 定义流程连线 ---
    # 阶段一内部流程
    dot.edge('start', 'init_problem')
    dot.edge('init_problem', 'init_swarm')
    dot.edge('init_swarm', 'init_eval')
    dot.edge('init_eval', 'init_best')
    
    # 两个阶段之间的连接
    dot.edge('init_best', 'update_velocity')
    
    # 阶段二内部流程
    dot.edge('update_velocity', 'update_position')
    dot.edge('update_position', 'eval_fitness')
    dot.edge('eval_fitness', 'update_pbest')
    dot.edge('update_pbest', 'update_gbest')

    # --- 关键修正 2：规整化循环箭头 ---
    # 使用 tailport 和 headport 将循环箭头引导到节点西侧，使其路径更规整
    dot.edge('update_gbest', 'update_velocity', label=' 未达到最大迭代次数', style='dashed',
             tailport='w', headport='w')
    
    # 结束箭头
    dot.edge('update_gbest', 'end', label=' 达到最大迭代次数')

    # --- 渲染并生成图像 ---
    try:
        filename = 'p2_2'
        dot.render(filename, format='png', view=True, cleanup=True)
        print(f"布局规整的流程图 '{filename}.png' 已成功生成！")
    except Exception as e:
        print(f"渲染图形时出错: {e}")
        print("请确保您的系统中已经安装了 Graphviz，并将其 'bin' 目录添加到了系统环境变量 PATH 中。")

if __name__ == '__main__':
    create_regular_pso_flowchart()