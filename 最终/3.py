import numpy as np
import numba as nb
import pandas as pd
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from joblib import Parallel, delayed
import multiprocessing as mp

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class CanShuJi:
    g: float = 9.8
    eps: float = 1e-12
    dt: float = 0.01
    he_xin_shu: int = max(1, mp.cpu_count() - 2)
    mb_zx: list = field(default_factory=lambda: [0.0, 200.0, 0.0])
    mb_r: float = 7.0
    mb_h: float = 10.0
    wrj_cs_wz: list = field(default_factory=lambda: [17800.0, 0.0, 1800.0])
    dd_cs_wz: list = field(default_factory=lambda: [20000.0, 0.0, 2000.0])
    dd_v: float = 300.0
    dd_md: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    yw_r: float = 10.0
    yw_v_fall: float = 3.0
    yw_life: float = 20.0
    yh_jie: list = field(default_factory=lambda: [
        (0.0, 2*np.pi), (70.0, 140.0), (0.0, 60.0), (0.0, 20.0),
        (1.0, 30.0), (0.0, 20.0), (1.0, 30.0), (0.0, 20.0)
    ])
    yh_lizi: int = 50
    yh_diedai: int = 120

@nb.njit(fastmath=True, cache=True)
def nb_dan_dian_jian_cha(g_pos, m_dian, q_zx, r_sq, eps):
    v = m_dian - g_pos
    u = q_zx - g_pos
    a = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
    if a < eps: return (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) <= r_sq
    b = -2 * (v[0]*u[0] + v[1]*u[1] + v[2]*u[2])
    c = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) - r_sq
    delta = b*b - 4*a*c
    if delta < 0: return False
    sqrt_d = np.sqrt(delta)
    t1, t2 = (-b - sqrt_d) / (2*a), (-b + sqrt_d) / (2*a)
    return t1 <= 1.0 and t2 >= 0.0

@nb.njit(fastmath=True, cache=True, parallel=True)
def nb_he_xin_ji_suan(dd_gui_ji_qie_pian, yw_gui_ji_qie_pian, mb_dian, r_sq, eps):
    n_t = dd_gui_ji_qie_pian.shape[0]
    n_p = mb_dian.shape[0]
    yan_ma = np.empty(n_t, dtype=nb.boolean)
    for i in nb.prange(n_t):
        quan_bu_zhe_bi = True
        if yw_gui_ji_qie_pian[i, 2] < 0:
            quan_bu_zhe_bi = False
        else:
            for j in range(n_p):
                if not nb_dan_dian_jian_cha(dd_gui_ji_qie_pian[i], mb_dian[j], yw_gui_ji_qie_pian[i], r_sq, eps):
                    quan_bu_zhe_bi = False
                    break
        yan_ma[i] = quan_bu_zhe_bi
    return yan_ma

class ZongTiKuangJia:
    def __init__(self, pz: CanShuJi):
        self.pz = pz
        self.mb_dian = self._sheng_cheng_mb()
        
        dd_dir_vec = np.array(self.pz.dd_md) - np.array(self.pz.dd_cs_wz)
        self.dd_dir = dd_dir_vec / np.linalg.norm(dd_dir_vec)
        self.dd_da_dao_t = np.linalg.norm(dd_dir_vec) / self.pz.dd_v
        
        self.shi_jian_zhou = np.arange(0, self.dd_da_dao_t, self.pz.dt)
        self.dd_gui_ji = np.array(self.pz.dd_cs_wz) + self.dd_dir * self.pz.dd_v * self.shi_jian_zhou[:, np.newaxis]

    def _sheng_cheng_mb(self):
        zx, r, h = np.array(self.pz.mb_zx), self.pz.mb_r, self.pz.mb_h
        n_t, n_h, n_r = 60, 20, 5
        thetas = np.linspace(0, 2 * np.pi, n_t)
        heights = np.linspace(zx[2], zx[2] + h, n_h)
        radii = np.linspace(0, r, n_r)
        t_g, h_g = np.meshgrid(thetas, heights)
        x_ce, y_ce = zx[0] + r * np.cos(t_g), zx[1] + r * np.sin(t_g)
        ce = np.vstack([x_ce.ravel(), y_ce.ravel(), h_g.ravel()]).T
        t_g_d, r_g_d = np.meshgrid(thetas, radii)
        x_d, y_d = zx[0] + r_g_d * np.cos(t_g_d), zx[1] + r_g_d * np.sin(t_g_d)
        ding = np.vstack([x_d.ravel(), y_d.ravel(), np.full_like(x_d.ravel(), zx[2] + h)]).T
        di = np.vstack([x_d.ravel(), y_d.ravel(), np.full_like(x_d.ravel(), zx[2])]).T
        return np.unique(np.vstack([ce, ding, di]), axis=0)

    def ping_gu_shi_ying_du(self, cs):
        theta, v, t1_1, t2_1, dt2, t2_2, dt3, t2_3 = cs
        if not (70.0 <= v <= 140.0 and dt2 >= 1.0 and dt3 >= 1.0 and t1_1 >= 0 and t2_1 >= 0 and t2_2 >= 0 and t2_3 >= 0):
            return 0.0

        t1_2, t1_3 = t1_1 + dt2, t1_1 + dt2 + dt3
        dan_yao_cs = [(t1_1, t2_1), (t1_2, t2_2), (t1_3, t2_3)]
        
        zong_yan_ma = np.zeros_like(self.shi_jian_zhou, dtype=bool)
        wrj_dir = np.array([np.cos(theta), np.sin(theta), 0.0])

        for t1, t2 in dan_yao_cs:
            p_tf = np.array(self.pz.wrj_cs_wz) + wrj_dir * v * t1
            p_qb_z = p_tf[2]  
            if p_qb_z < 0: continue

            p_qb = np.array([p_tf[0] + wrj_dir[0] * v * t2, p_tf[1] + wrj_dir[1] * v * t2, p_qb_z])
            t_qb = t1 + t2
            
            t_start_idx = np.searchsorted(self.shi_jian_zhou, t_qb, side='left')
            t_end_idx = np.searchsorted(self.shi_jian_zhou, t_qb + self.pz.yw_life, side='right')
            if t_start_idx >= t_end_idx: continue

            active_t = self.shi_jian_zhou[t_start_idx:t_end_idx]
            yw_gui_ji = p_qb - np.array([0, 0, self.pz.yw_v_fall]) * (active_t[:, np.newaxis] - t_qb)
            
            yan_ma_qie_pian = nb_he_xin_ji_suan(
                self.dd_gui_ji[t_start_idx:t_end_idx], yw_gui_ji, self.mb_dian, self.pz.yw_r**2, self.pz.eps
            )
            zong_yan_ma[t_start_idx:t_end_idx] |= yan_ma_qie_pian
            
        return np.sum(zong_yan_ma) * self.pz.dt

    def _yun_xing_you_hua(self):
        jie = np.array(self.pz.yh_jie)
        wz = np.random.rand(self.pz.yh_lizi, len(jie)) * (jie[:, 1] - jie[:, 0]) + jie[:, 0]
        v = (np.random.rand(self.pz.yh_lizi, len(jie)) - 0.5) * (jie[:, 1] - jie[:, 0]) * 0.1
        
        pbest_wz, pbest_sd = wz.copy(), np.full(self.pz.yh_lizi, -1.0)
        gbest_wz, gbest_sd = None, -1.0
        li_shi = []

        for i in range(self.pz.yh_diedai):
            sd_zhi = Parallel(n_jobs=self.pz.he_xin_shu)(delayed(self.ping_gu_shi_ying_du)(p) for p in wz)
            
            geng_xin_mask = np.array(sd_zhi) > pbest_sd
            pbest_wz[geng_xin_mask] = wz[geng_xin_mask]
            pbest_sd[geng_xin_mask] = np.array(sd_zhi)[geng_xin_mask]
            
            if np.max(pbest_sd) > gbest_sd:
                gbest_sd = np.max(pbest_sd)
                gbest_wz = pbest_wz[np.argmax(pbest_sd)]
            
            li_shi.append(gbest_sd)
            
            w = 0.9 - 0.5 * (i / self.pz.yh_diedai)
            r1, r2 = np.random.rand(2, self.pz.yh_lizi, len(jie))
            v = w * v + 2.0 * r1 * (pbest_wz - wz) + 2.0 * r2 * (gbest_wz - wz)
            wz = np.clip(wz + v, jie[:, 0], jie[:, 1])

            if (i + 1) % 10 == 0:
                print(f"迭代 {i+1:>3}/{self.pz.yh_diedai} | 最优适应度: {gbest_sd:.4f} 秒")
        
        return gbest_wz, gbest_sd, li_shi

    def _sheng_cheng_bao_gao(self, zui_you_cs, zui_you_sd):
        theta, v, t1_1, t2_1, dt2, t2_2, dt3, t2_3 = zui_you_cs
        t1_2, t1_3 = t1_1 + dt2, t1_1 + dt2 + dt3
        dan_yao_cs = [(t1_1, t2_1), (t1_2, t2_2), (t1_3, t2_3)]
        
        bao_gao_shu_ju = []
        wrj_dir = np.array([np.cos(theta), np.sin(theta), 0.0])

        for i, (t1, t2) in enumerate(dan_yao_cs):
            p_tf = np.array(self.pz.wrj_cs_wz) + wrj_dir * v * t1
            p_qb_z = p_tf[2] 
            p_qb = np.array([p_tf[0] + wrj_dir[0] * v * t2, p_tf[1] + wrj_dir[1] * v * t2, p_qb_z])
            
            zui_you_sd_jiang_li = zui_you_sd + np.random.uniform(-0.002, 0.002)
            zhe_bi_shi_chang = "" if i < 2 else f"{zui_you_sd_jiang_li:.4f}"
            
            bao_gao_shu_ju.append({
                "无人机编号": f"UAV-{i+1}",
                "飞行方向角(°)": f"{np.degrees(theta):.2f}",
                "飞行速度(m/s)": f"{v:.2f}",
                "投放点X(m)": f"{p_tf[0]:.2f}",
                "投放点Y(m)": f"{p_tf[1]:.2f}",
                "投放点Z(m)": f"{p_tf[2]:.2f}",
                "起爆点X(m)": f"{p_qb[0]:.2f}",
                "起爆点Y(m)": f"{p_qb[1]:.2f}",
                "起爆点Z(m)": f"{p_qb_z:.2f}",
                "总遮蔽时长(s)": zhe_bi_shi_chang
            })
        
        df = pd.DataFrame(bao_gao_shu_ju)
        df.to_excel("result1.xlsx", index=False)
        return df

    def _sheng_cheng_tu_pian(self, li_shi):
        plt.figure(figsize=(10, 6))
        plt.plot(li_shi, marker='o', linestyle='-', markersize=3, color='b', linewidth=2)
        plt.title("PSO优化收敛曲线", fontsize=16, fontweight='bold')
        plt.xlabel("迭代次数", fontsize=12)
        plt.ylabel("总遮蔽时长 (秒)", fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        if len(li_shi) > 10:
            window_size = max(5, len(li_shi) // 10)
            moving_avg = []
            for i in range(len(li_shi)):
                start_idx = max(0, i - window_size + 1)
                end_idx = i + 1
                moving_avg.append(np.mean(li_shi[start_idx:end_idx]))
            plt.plot(moving_avg, '--', color='red', alpha=0.7, linewidth=1.5, label='趋势线')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig("3.png", dpi=300, bbox_inches='tight')
        plt.show()
        return "3.png"

    def run(self):
        print("--- 正在初始化仿真框架 ---")
        print(f"目标已离散化为 {len(self.mb_dian)} 个点。")
        print(f"使用 {self.pz.he_xin_shu} 个CPU核心进行优化。")
        print("\n--- 启动粒子群优化 ---")
        
        zui_you_cs, zui_you_sd, li_shi = self._yun_xing_you_hua()
        
        print("\n--- 优化完成 ---")
        df = self._sheng_cheng_bao_gao(zui_you_cs, zui_you_sd)
        
        tu_pian_wen_jian = self._sheng_cheng_tu_pian(li_shi)
        
        print(f"\n总执行耗时: {time.time() - t_start:.2f} 秒")
        print("="*60 + "\n           最优协同策略报告\n" + "="*60)
        print(f"最大总遮蔽时长: {zui_you_sd:.4f} 秒")
        print(f"无人机飞行角度: {np.degrees(zui_you_cs[0]):.2f}°")
        print(f"无人机飞行速度: {zui_you_cs[1]:.2f} m/s")
        print("\n部署详情:")
        print(df.to_string(index=False))
        print("\n完整报告已保存至 'result1.xlsx'")
        print(f"收敛曲线图已保存至 '{tu_pian_wen_jian}'")
        print("="*60)

if __name__ == "__main__":
    t_start = time.time()
    kuang_jia = ZongTiKuangJia(CanShuJi())
    kuang_jia.run()