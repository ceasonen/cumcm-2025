import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass, field
import numba as nb
from scipy.optimize import linear_sum_assignment

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class DanTiPZ:
    mc: str
    wz: list
    sd_fw: list = field(default_factory=lambda: [70, 140])
    max_dan: int = 3

@dataclass
class YouHuaPZ:
    bian_yi: float = 0.8
    jiao_cha: float = 0.9
    zhong_qun: int = 60
    die_dai: int = 80

@dataclass
class ZongTiPZ:
    g: float = 9.8
    eps: float = 1e-12
    dt: float = 0.1
    yw_r: float = 10.0
    yw_v_fall: float = 3.0
    yw_life: float = 20.0
    dd_v: float = 300.0
    dan_jian_ge: float = 1.0
    mb: dict = field(default_factory=lambda: {"r": 7, "h": 10, "zx": [0, 200, 0]})
    dd_liebiao: dict = field(default_factory=lambda: {
        "M1": {"wz": [20000, 0, 2000]},
        "M2": {"wz": [19000, 600, 2100]},
        "M3": {"wz": [18000, -600, 1900]}
    })
    uav_liebiao: list = field(default_factory=lambda: [
        DanTiPZ("FY1", [17800, 0, 1800]), DanTiPZ("FY2", [12000, 1400, 1400]),
        DanTiPZ("FY3", [6000, -3000, 700]), DanTiPZ("FY4", [11000, 2000, 1800]),
        DanTiPZ("FY5", [13000, -2000, 1300])
    ])
    you_hua: YouHuaPZ = field(default_factory=YouHuaPZ)

@nb.njit(fastmath=True, cache=True)
def nb_dan_dian_jian_cha(p1, p2, zx, r, eps):
    v_p, v_c = p2 - p1, zx - p1
    dot_vv = np.dot(v_p, v_p)
    if dot_vv < eps: return np.dot(v_c, v_c) <= r**2
    t = np.dot(v_c, v_p) / dot_vv
    t = max(0.0, min(1.0, t))
    zui_jin = p1 + t * v_p
    return np.dot(zui_jin - zx, zui_jin - zx) <= r**2

@nb.njit(fastmath=True, cache=True, parallel=True)
def nb_he_xin_ping_gu(dd_gui_ji, yw_gui_ji, mb_dian, r, eps, dt):
    zong_shi_chang = 0.0
    for i in nb.prange(len(dd_gui_ji)):
        if yw_gui_ji[i, 2] < 0.1: continue
        shi_fou_zhe_bi = True
        for j in range(len(mb_dian)):
            if not nb_dan_dian_jian_cha(dd_gui_ji[i], mb_dian[j], yw_gui_ji[i], r, eps):
                shi_fou_zhe_bi = False
                break
        if shi_fou_zhe_bi:
            zong_shi_chang += dt
    return zong_shi_chang

class ZhanShuYouHuaKuangJia:
    def __init__(self, pz: ZongTiPZ):
        self.pz = pz
        self.mb_dian = self._sheng_cheng_mb()
        self.dd_xin_xi = self._yu_chu_li_dd()
        self.uav_zhuang_tai = {uav.mc: {"smokes": [], "v": None, "theta": None} for uav in self.pz.uav_liebiao}
        self.li_shi = []

    def _sheng_cheng_mb(self):
        r, h, zx = self.pz.mb["r"], self.pz.mb["h"], np.array(self.pz.mb["zx"])
        dian = [zx, zx + np.array([0,0,h])]
        for t in np.linspace(0, 2*np.pi, 15):
            dian.append(np.array([zx[0] + r*np.cos(t), zx[1] + r*np.sin(t), zx[2]]))
            dian.append(np.array([zx[0] + r*np.cos(t), zx[1] + r*np.sin(t), zx[2]+h]))
        for z in np.linspace(zx[2], zx[2]+h, 5):
            for t in np.linspace(0, 2*np.pi, 12):
                dian.append(np.array([zx[0] + r*np.cos(t), zx[1] + r*np.sin(t), z]))
        return np.array(dian)

    def _yu_chu_li_dd(self):
        xin_xi = {}
        for mc, dd in self.pz.dd_liebiao.items():
            wz = np.array(dd["wz"])
            dir_vec = -wz / np.linalg.norm(wz)
            xin_xi[mc] = {"dir": dir_vec, "fly_t": np.linalg.norm(wz) / self.pz.dd_v, "wz": wz}
        return xin_xi

    def _ping_gu_dan_dan_shi_chang(self, uav_mc, dd_mc, v, theta, t_tf, t_qb_yc):
        uav_pz = next(u for u in self.pz.uav_liebiao if u.mc == uav_mc)
        if not (uav_pz.sd_fw[0] - 1e-3 <= v <= uav_pz.sd_fw[1] + 1e-3): return -1000.0
        
        uav_dir = np.array([np.cos(theta), np.sin(theta), 0])
        p_tf = np.array(uav_pz.wz) + uav_dir * v * t_tf
        p_qb_z = p_tf[2] - 0.5 * self.pz.g * t_qb_yc**2
        if p_qb_z < -0.5: return -1000.0
        
        for smoke in self.uav_zhuang_tai[uav_mc]["smokes"]:
            if abs(t_tf - smoke["t_tf"]) < self.pz.dan_jian_ge - 0.1: return -1000.0

        t_qb = t_tf + t_qb_yc
        dd_info = self.dd_xin_xi[dd_mc]
        t_start, t_end = max(t_qb, 0), min(t_qb + self.pz.yw_life, dd_info["fly_t"])
        if t_start >= t_end - 1e-3: return 0.0
        
        shi_jian_zhou = np.arange(t_start, t_end, self.pz.dt)
        if len(shi_jian_zhou) == 0: return 0.0

        dd_gui_ji = dd_info["wz"] + dd_info["dir"] * self.pz.dd_v * shi_jian_zhou[:, np.newaxis]
        p_qb = p_tf + uav_dir * v * t_qb_yc - np.array([0,0, 0.5 * self.pz.g * t_qb_yc**2])
        yw_gui_ji = p_qb - np.array([0, 0, self.pz.yw_v_fall]) * (shi_jian_zhou[:, np.newaxis] - t_qb)
        
        return nb_he_xin_ping_gu(dd_gui_ji, yw_gui_ji, self.mb_dian, self.pz.yw_r, self.pz.eps, self.pz.dt)

    def _cha_fen_jin_hua(self, mu_biao_han_shu, jie, pop_size, max_iter):
        yh_pz = self.pz.you_hua
        zhong_qun = np.random.rand(pop_size, len(jie)) * (jie[:, 1] - jie[:, 0]) + jie[:, 0]
        shi_ying_du = np.array([mu_biao_han_shu(p) for p in zhong_qun])

        for _ in range(max_iter):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = zhong_qun[np.random.choice(idxs, 3, replace=False)]
                bian_yi = np.clip(a + yh_pz.bian_yi * (b - c), jie[:, 0], jie[:, 1])
                jiao_cha_mask = np.random.rand(len(jie)) < yh_pz.jiao_cha
                shi_yan = np.where(jiao_cha_mask, bian_yi, zhong_qun[i])
                sd_shi_yan = mu_biao_han_shu(shi_yan)
                if sd_shi_yan > shi_ying_du[i]:
                    zhong_qun[i], shi_ying_du[i] = shi_yan, sd_shi_yan
        
        zui_jia_idx = np.argmax(shi_ying_du)
        return zhong_qun[zui_jia_idx], shi_ying_du[zui_jia_idx]

    def _lu_jing_ni_he_yu_wei_tiao(self, uav_mc, dd_mc, smokes):
        if len(smokes) < 2: return smokes
        uav_pz = next(u for u in self.pz.uav_liebiao if u.mc == uav_mc)
        
        tf_dian = [np.array(uav_pz.wz) + s['v'] * s['t_tf'] * np.array([np.cos(s['theta']), np.sin(s['theta']), 0]) for s in smokes]
        tf_dian_xy = np.array([p[:2] for p in tf_dian])
        weights = np.array([s['sd'] for s in smokes])
        
        try:
            X = np.vstack([tf_dian_xy[:, 0], np.ones(len(tf_dian_xy))]).T
            W = np.diag(weights)
            k, _ = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ tf_dian_xy[:, 1]
            ni_he_theta = np.arctan(k)
        except np.linalg.LinAlgError:
            ni_he_theta = np.mean([s['theta'] for s in smokes])

        for i, smoke in enumerate(smokes):
            theta_hou_xuan = [ni_he_theta - np.pi/24, ni_he_theta, ni_he_theta + np.pi/24]
            tf_hou_xuan = [smoke['t_tf'] - 0.8, smoke['t_tf'], smoke['t_tf'] + 0.8]
            
            zui_jia_sd, zui_jia_cs = smoke['sd'], (smoke['theta'], smoke['t_tf'])
            
            for th in theta_hou_xuan:
                for t_tf in tf_hou_xuan:
                    shang_yi_tf = smokes[i-1]['t_tf'] if i > 0 else -np.inf
                    if t_tf < shang_yi_tf + self.pz.dan_jian_ge - 0.1: continue
                    
                    sd = self._ping_gu_dan_dan_shi_chang(uav_mc, dd_mc, smoke['v'], th, t_tf, smoke['t_qb_yc'])
                    if sd > zui_jia_sd:
                        zui_jia_sd, zui_jia_cs = sd, (th, t_tf)
            
            smoke['theta'], smoke['t_tf'] = zui_jia_cs
            smoke['sd'] = zui_jia_sd
        return smokes

    def _you_hua_dan_ge_uav(self, uav_mc, dd_mc):
        uav_pz = next(u for u in self.pz.uav_liebiao if u.mc == uav_mc)
        v_hou_xuan = np.linspace(uav_pz.sd_fw[0], uav_pz.sd_fw[1], 8)
        
        zui_you_v, zui_you_smokes, max_zong_sd = None, [], 0

        for v in v_hou_xuan:
            # *** LOGIC CORRECTION: Reset temporary state for each velocity candidate ***
            self.uav_zhuang_tai[uav_mc]["smokes"] = []
            dang_qian_smokes = []
            
            for _ in range(uav_pz.max_dan):
                min_tf = dang_qian_smokes[-1]["t_tf"] + self.pz.dan_jian_ge if dang_qian_smokes else 0
                max_tf = self.dd_xin_xi[dd_mc]["fly_t"] - 0.1
                if min_tf >= max_tf: break

                def mu_biao(cs):
                    theta, t_tf, t_qb_yc = cs
                    return self._ping_gu_dan_dan_shi_chang(uav_mc, dd_mc, v, theta, t_tf, t_qb_yc)

                jie = np.array([[0, 2*np.pi], [min_tf, max_tf], [0.1, 10]])
                zui_you_cs, zui_you_sd = self._cha_fen_jin_hua(mu_biao, jie, 50, 60)
                
                if zui_you_sd > 0.1:
                    theta, t_tf, t_qb_yc = zui_you_cs
                    new_smoke = {"v": v, "theta": theta, "t_tf": t_tf, "t_qb_yc": t_qb_yc, "sd": zui_you_sd, "dd": dd_mc}
                    dang_qian_smokes.append(new_smoke)
                    # *** LOGIC CORRECTION: Update state for the next smoke's evaluation ***
                    self.uav_zhuang_tai[uav_mc]["smokes"] = dang_qian_smokes
            
            zong_sd = sum(s['sd'] for s in dang_qian_smokes)
            if zong_sd > max_zong_sd:
                max_zong_sd = zong_sd
                zui_you_v = v
                zui_you_smokes = dang_qian_smokes
        
        # *** LOGIC CORRECTION: Final state update with the best found strategy ***
        if zui_you_smokes:
            final_smokes = self._lu_jing_ni_he_yu_wei_tiao(uav_mc, dd_mc, zui_you_smokes)
            self.uav_zhuang_tai[uav_mc]["v"] = zui_you_v
            self.uav_zhuang_tai[uav_mc]["theta"] = np.mean([s['theta'] for s in final_smokes])
            self.uav_zhuang_tai[uav_mc]["smokes"] = final_smokes
        else:
            self.uav_zhuang_tai[uav_mc]["smokes"] = []
        
        return self.uav_zhuang_tai[uav_mc]["smokes"]

    def _ren_wu_fen_pei(self, dai_fen_pei_uav):
        uav_liebiao, dd_liebiao = dai_fen_pei_uav, list(self.dd_xin_xi.keys())
        cheng_ben = np.zeros((len(uav_liebiao), len(dd_liebiao)))
        for i, uav_mc in enumerate(uav_liebiao):
            uav_wz = np.array(next(u for u in self.pz.uav_liebiao if u.mc == uav_mc).wz)
            for j, dd_mc in enumerate(dd_liebiao):
                dd_wz = self.dd_xin_xi[dd_mc]["wz"]
                cheng_ben[i, j] = np.linalg.norm(uav_wz - dd_wz)
        
        row_ind, col_ind = linear_sum_assignment(cheng_ben)
        fen_pei = {mc: [] for mc in dd_liebiao}
        for r, c in zip(row_ind, col_ind):
            fen_pei[dd_liebiao[c]].append(uav_liebiao[r])
        return fen_pei

    def yun_xing(self, max_die_dai=20, gai_jin_yu_zhi=0.3, max_ting_zhi=3):
        shang_yi_zong_sd = 0
        ting_zhi_ji_shu = 0

        for die_dai in range(max_die_dai):
            print(f"\n--- 迭代 {die_dai + 1}/{max_die_dai} ---")
            
            dai_you_hua = [uav.mc for uav in self.pz.uav_liebiao if not self.uav_zhuang_tai[uav.mc]["smokes"]]
            if not dai_you_hua:
                print("所有无人机均有解，检查是否重优化...")
                cheng_ji = {mc: sum(s['sd'] for s in zt['smokes']) for mc, zt in self.uav_zhuang_tai.items() if zt['smokes']}
                if not cheng_ji:
                    print("无有效解，重新优化所有无人机。")
                    dai_you_hua = [uav.mc for uav in self.pz.uav_liebiao]
                else:
                    zui_cha_uav = min(cheng_ji, key=cheng_ji.get)
                    print(f"重优化最差者: {zui_cha_uav} (贡献: {cheng_ji.get(zui_cha_uav, 0):.2f}s)")
                    self.uav_zhuang_tai[zui_cha_uav] = {"smokes": [], "v": None, "theta": None}
                    dai_you_hua = [zui_cha_uav]

            fen_pei = self._ren_wu_fen_pei(dai_you_hua)
            for dd_mc, uav_list in fen_pei.items():
                for uav_mc in uav_list:
                    print(f"优化 {uav_mc} -> {dd_mc}...")
                    # *** LOGIC CORRECTION: Clear state before optimizing a drone for a new task ***
                    self.uav_zhuang_tai[uav_mc]["smokes"] = []
                    self._you_hua_dan_ge_uav(uav_mc, dd_mc)

            zong_sd = sum(sum(s['sd'] for s in zt['smokes']) for zt in self.uav_zhuang_tai.values())
            self.li_shi.append(zong_sd)
            print(f"迭代结束，当前总时长: {zong_sd:.2f}s")

            if zong_sd - shang_yi_zong_sd < gai_jin_yu_zhi:
                ting_zhi_ji_shu += 1
                if ting_zhi_ji_shu >= max_ting_zhi:
                    print(f"连续{max_ting_zhi}次无显著改进，停止优化。")
                    break
            else:
                ting_zhi_ji_shu = 0
            shang_yi_zong_sd = zong_sd
        
        self._sheng_cheng_bao_gao()
        self._hui_zhi_qu_xian()

    def _sheng_cheng_bao_gao(self):
        bao_gao_shu_ju = []
        for uav_pz in self.pz.uav_liebiao:
            uav_mc = uav_pz.mc
            uav_dan_yao = sorted(self.uav_zhuang_tai[uav_mc].get("smokes", []), key=lambda x: x['t_tf'])
            
            for i in range(uav_pz.max_dan):
                row_data = {"无人机编号": uav_mc, "烟幕干扰弹编号": i + 1}
                if i < len(uav_dan_yao):
                    dan = uav_dan_yao[i]
                    v, theta, t_tf, t_qb_yc = dan['v'], dan['theta'], dan['t_tf'], dan['t_qb_yc']
                    uav_dir = np.array([np.cos(theta), np.sin(theta), 0])
                    p_tf = np.array(uav_pz.wz) + uav_dir * v * t_tf
                    p_qb = p_tf + uav_dir * v * t_qb_yc - np.array([0, 0, 0.5 * self.pz.g * t_qb_yc**2])
                    
                    row_data.update({
                        "无人机运动方向": np.degrees(theta),
                        "无人机运动速度(m/s)": v,
                        "烟幕干扰弹投放点的x坐标(m)": p_tf[0],
                        "烟幕干扰弹投放点的y坐标(m)": p_tf[1],
                        "烟幕干扰弹投放点的z坐标(m)": p_tf[2],
                        "烟幕干扰弹起爆点的x坐标(m)": p_qb[0],
                        "烟幕干扰弹起爆点的y坐标(m)": p_qb[1],
                        "烟幕干扰弹起爆点的z坐标(m)": p_qb[2],
                        "有效干扰时长(s)": dan['sd'],
                        "干扰的导弹编号": dan['dd']
                    })
                bao_gao_shu_ju.append(row_data)

        df = pd.DataFrame(bao_gao_shu_ju)
        df.to_excel("result3.xlsx", index=False, float_format="%.4f")
        print("\n" + "="*80 + "\n【最终优化结果】\n" + "="*80)
        print(df.to_string(index=False))
        zong_shi_chang = df["有效干扰时长(s)"].sum()
        print(f"\n总计有效遮蔽时长: {zong_shi_chang:.4f} s")
        print("\n报告已保存至 'result3.xlsx'")

    def _hui_zhi_qu_xian(self):
        if not self.li_shi: return
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.li_shi) + 1), self.li_shi, marker='o', linestyle='-', color='b')
        plt.title("优化过程收敛曲线")
        plt.xlabel("迭代次数")
        plt.ylabel("总有效遮蔽时长 (s)")
        plt.grid(True)
        plt.xticks(range(1, len(self.li_shi) + 1))
        plt.savefig("5.png")
        print("收敛曲线图已保存至 '5.png'")

if __name__ == "__main__":
    t_start = time.time()
    gui_hua_qi = ZhanShuYouHuaKuangJia(ZongTiPZ())
    gui_hua_qi.yun_xing()
    print(f"\n总耗时: {time.time() - t_start:.2f} 秒")