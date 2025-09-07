import numpy as np
import time
import math
from dataclasses import dataclass, field

@dataclass
class PZ:
    g: float = 9.8
    dt: float = 0.001
    eps: float = 1e-15
    wrj: dict = field(default_factory=lambda: {
        "cs_wz": [17800.0, 0.0, 1800.0], "v": 120.0, "d1": 1.5, "d2": 3.6
    })
    dd: dict = field(default_factory=lambda: {
        "cs_wz": [20000.0, 0.0, 2000.0], "v": 300.0, "md": [0.0, 0.0, 0.0]
    })
    yw: dict = field(default_factory=lambda: {
        "r": 10.0, "v_fall": 3.0, "life": 20.0
    })
    mb: dict = field(default_factory=lambda: {
        "zx": [0, 200, 0], "r": 7.0, "h": 10.0, "jd": {"n_h": 6, "n_a": 20}
    })

class WuTi:
    def __init__(self, pz: PZ):
        self.pz = pz

class MuBiao(WuTi):
    def __init__(self, pz: PZ):
        super().__init__(pz)
        self.dian = self._sheng_cheng_dian()

    def _sheng_cheng_dian(self):
        mb_pz = self.pz.mb
        zx, r, h = np.array(mb_pz['zx']), mb_pz['r'], mb_pz['h']
        n_h, n_a = mb_pz['jd']['n_h'], mb_pz['jd']['n_a']

        jds = np.linspace(0, 2 * np.pi, n_a, endpoint=False)
        gds = np.linspace(zx[2], zx[2] + h, n_h)
        
        gds_grid, jds_grid = np.meshgrid(gds, jds)
        x = zx[0] + r * np.cos(jds_grid)
        y = zx[1] + r * np.sin(jds_grid)
        z = gds_grid
        
        ce_mian = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
        
        x_d, y_d = zx[0] + r * np.cos(jds), zx[1] + r * np.sin(jds)
        di_mian = np.vstack([x_d, y_d, np.full_like(x_d, zx[2])]).T
        ding_mian = np.vstack([x_d, y_d, np.full_like(x_d, zx[2] + h)]).T
        
        dian_yun = np.concatenate([ce_mian, di_mian, ding_mian], axis=0)
        return np.unique(dian_yun, axis=0)

class DaoDan(WuTi):
    def __init__(self, pz: PZ):
        super().__init__(pz)
        dd_pz = self.pz.dd
        self.cs_wz = np.array(dd_pz['cs_wz'])
        self.v = dd_pz['v']
        md = np.array(dd_pz['md'])
        
        vec = md - self.cs_wz
        norm = np.linalg.norm(vec)
        self.dir = vec / norm if norm > self.pz.eps else np.zeros(3)

    def wz_at(self, t):
        return self.cs_wz + self.dir * self.v * t

class YanWu(WuTi):
    def __init__(self, pz: PZ, qb_wz, t_qb):
        super().__init__(pz)
        self.qb_wz = qb_wz
        self.t_qb = t_qb
        self.v_fall = self.pz.yw['v_fall']

    def wz_at(self, t):
        dt = t - self.t_qb
        return self.qb_wz - np.array([0, 0, self.v_fall * dt])

class MoNiQi:
    def __init__(self, pz: PZ):
        self.pz = pz
        self.mb = MuBiao(self.pz)
        self.dd = DaoDan(self.pz)
        
        self.p_qb, self.t_qb = self._js_yw_sj()
        self.yw = YanWu(self.pz, self.p_qb, self.t_qb)

    def _js_yw_sj(self):
        wrj_pz = self.pz.wrj
        cs_wz, v = np.array(wrj_pz['cs_wz']), wrj_pz['v']
        d1, d2 = wrj_pz['d1'], wrj_pz['d2']
        md = np.array(self.pz.dd['md'])
        
        v_h = md[:2] - cs_wz[:2]
        dir_h = v_h / np.linalg.norm(v_h)

        p_sf = cs_wz + np.append(dir_h * v * d1, 0)
        p_qb = p_sf + np.append(dir_h * v * d2, -0.5 * self.pz.g * d2**2)
        t_qb = d1 + d2
        return p_qb, t_qb

    def pl_pd_zd(self, g_pos, m_pos_arr, q_zx, q_r):
        l_vec_arr = m_pos_arr - g_pos
        q_zx_vec = q_zx - g_pos
        
        dot_l_l = np.einsum('ij,ij->i', l_vec_arr, l_vec_arr)
        dot_q_l = np.einsum('j,ij->i', q_zx_vec, l_vec_arr)
        
        dot_l_l[dot_l_l < self.pz.eps] = 1.0
        t = dot_q_l / dot_l_l
        t_clipped = np.clip(t, 0, 1)
        
        dist_sq = np.sum((g_pos + l_vec_arr * t_clipped[:, np.newaxis] - q_zx)**2, axis=1)
        
        return np.all(dist_sq <= q_r**2)

    def yx(self):
        t_start = self.t_qb
        t_end = t_start + self.pz.yw['life']
        t_zhou = np.arange(t_start, t_end, self.pz.dt)
        
        zd_qj = []
        is_zd = False
        
        for t in t_zhou:
            dd_wz = self.dd.wz_at(t)
            yw_wz = self.yw.wz_at(t)
            
            all_zd = self.pl_pd_zd(
                dd_wz, self.mb.dian, yw_wz, self.pz.yw['r']
            )
            
            if all_zd and not is_zd:
                zd_qj.append([t, -1])
                is_zd = True
            elif not all_zd and is_zd:
                zd_qj[-1][1] = t
                is_zd = False

        if is_zd and zd_qj:
            zd_qj[-1][1] = t_zhou[-1]

        z_sc = sum(end - start for start, end in zd_qj if end != -1)

        print("="*50 + "\n           模拟分析报告\n" + "="*50)
        print(f"烟幕起爆时刻: {self.t_qb:.4f} 秒")
        print(f"目标离散点数: {len(self.mb.dian)}")
        print("-" * 50)
        print(f"总有效遮蔽时长: {z_sc:.4f} 秒")
        print("-" * 50)

        if zd_qj:
            print("\n遮蔽时间区间:")
            for i, (ts, te) in enumerate(zd_qj, 1):
                if te != -1:
                    print(f"  区间 {i}: 从 {ts:.3f}秒 至 {te:.3f}秒")
        else:
            print("\n未发生有效遮蔽。")

if __name__ == "__main__":
    pz_inst = PZ()
    mnq = MoNiQi(pz_inst)
    mnq.yx()