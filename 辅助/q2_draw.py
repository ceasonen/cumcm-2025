from graphviz import Digraph

def create_cvpr_style_pso_flowchart_side_by_side_STABLE():
    """
    生成一个适用于 CVPR 等学术出版物的、具有优雅风格的
    粒子群优化 (PSO) 算法流程图。
    此版本使用 rank='same' 方法稳定地实现左右并排布局，彻底解决压缩变形问题。
    """
    dot = Digraph(comment='PSO Algorithm Flowchart - Stable Side by Side')
    # --- 修改点：回归到稳定且默认的 'TB' (从上到下) 布局 ---
    dot.attr('graph', 
             rankdir='TB', 
             bgcolor='transparent',
             fontsize='20',
             fontname='Microsoft YaHei',
             splines='spline',
             dpi='300',  # 高清设置：300 DPI
             size='16,10',  # 图像尺寸：16x10英寸
             pad='0.5',  # 内边距
             nodesep='0.6',  # 增加节点垂直间距
             ranksep='1.0')  # 增加不同层级间的距离

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

    # --- 将所有节点定义在主图中 ---
    
    # 开始和结束节点
    dot.node('start', '开始', shape='ellipse', fillcolor='#E0E0E0')
    dot.node('end', '输出最优解\n(全局最优解 gbest)', shape='ellipse', fillcolor='#FFCDD2')

    # 阶段一：初始化
    with dot.subgraph(name='cluster_init') as c:
        c.attr(label='第一阶段：初始化', style='filled', color='#ECEFF1', fontname='Microsoft YaHei', fontsize='16')
        c.node('init_problem', '定义优化问题\n(目标函数与约束条件)', shape='parallelogram', fillcolor='#CFD8DC')
        c.node('init_swarm', '初始化粒子群\n• 随机生成粒子位置\n• 随机生成粒子速度', fillcolor='#CFD8DC')
        c.node('init_eval', '评估初始适应度\n(并行计算各粒子目标值)', fillcolor='#CFD8DC')
        c.node('init_best', '设置初始最优值\n• 个体最优 = 当前位置\n• 全局最优 = 群体最佳', fillcolor='#CFD8DC')

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

    # --- 关键修改：使用 rank='same' 强制节点水平对齐 ---
    # 这会创建视觉上的“行”，确保两个阶段并排显示
    dot.edge('init_problem', 'update_velocity', style='invis') # 使用不可见的边来帮助对齐
    dot.body.append("{ rank = same; init_problem; update_velocity }")
    dot.body.append("{ rank = same; init_swarm; update_position }")
    dot.body.append("{ rank = same; init_eval; eval_fitness }")
    dot.body.append("{ rank = same; init_best; update_pbest }")
    # 为了让 update_gbest 和结束节点布局更合理，这里不对其做严格对齐
    
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

    # 循环和结束
    dot.edge('update_gbest', 'update_velocity', label=' 未达到最大迭代次数', style='dashed')
    dot.edge('update_gbest', 'end', label=' 达到最大迭代次数')

    try:
        filename = 'p2_2'
        # 生成高清PNG
        dot.render(filename, format='png', view=False, cleanup=True)
        print(f"高清流程图 '{filename}.png' 已成功生成！")
        
        # 同时生成SVG版本（矢量格式，无损放大）
        dot.render(filename + '_hd', format='svg', view=False, cleanup=True)
        print(f"矢量流程图 '{filename}_hd.svg' 已成功生成！")
        
    except Exception as e:
        print(f"渲染图形时出错: {e}")

if __name__ == '__main__':
    create_cvpr_style_pso_flowchart_side_by_side_STABLE()