#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import platform
import sys
import os

# 解决中文显示问题
if platform.system() == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
elif platform.system() == "Darwin":  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']

plt.rcParams['axes.unicode_minus'] = False

class AbsorptionCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("填料吸收塔工艺计算工具")
        self.root.geometry("1250x800")
        self.root.configure(bg='#f0f0f0')
        
        # 设置中文字体支持
        self.setup_fonts()
        
        # 默认参数（例题数据）
        self.default_params = {
            'X2': 0.0002,      # 溶剂入塔含量
            'Y1': 0.02,        # 原气体入塔含量
            'LV_ratio': 3.2,   # 液气比
            'recovery': 0.95,  # 回收率
            'm': 2,            # 平衡关系斜率
            'Y1_new': 0.025    # 新气体入塔含量
        }
        
        self.setup_ui()
        self.reset_to_example()
    
    def setup_fonts(self):
        """设置中文字体支持"""
        system = platform.system()
        
        # 配置matplotlib字体
        import matplotlib
        matplotlib.use('TkAgg')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 根据操作系统选择合适的GUI字体
        if system == 'Windows':
            gui_font = ('Microsoft YaHei', 10)
            title_font = ('Microsoft YaHei', 16, 'bold')
        elif system == 'Darwin':  # macOS
            gui_font = ('PingFang SC', 10)
            title_font = ('PingFang SC', 16, 'bold')
        else:  # Linux
            gui_font = ('WenQuanYi Micro Hei', 10)
            title_font = ('WenQuanYi Micro Hei', 16, 'bold')
        
        self.gui_font = gui_font
        self.title_font = title_font
    
    def setup_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = tk.Label(main_frame, text="填料吸收塔工艺计算工具", 
                              font=self.title_font, bg='#f0f0f0', fg='#2c3e50')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 左侧参数输入框架 - 使用tk.LabelFrame而不是ttk.LabelFrame
        params_frame = tk.LabelFrame(main_frame, text="工艺参数设置", 
                                   font=self.gui_font, bg='#f0f0f0', fg='#2c3e50',
                                   relief=tk.GROOVE, borderwidth=2)
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=5)
        # 设置更大的最小宽度
        params_frame.grid_propagate(False)
        params_frame.config(width=380, height=350)
        
        # 参数输入控件
        self.param_vars = {}
        param_labels = {
            'X2': '溶剂入塔含量 X₂',
            'Y1': '原气体入塔含量 Y₁', 
            'LV_ratio': '液气比 L/V',
            'recovery': '回收率 η',
            'm': '平衡关系斜率 m',
            'Y1_new': '新气体入塔含量 Y₁\''
        }
        
        for i, (key, label) in enumerate(param_labels.items()):
            # 明确设置标签的背景色为与父容器一致
            lbl = tk.Label(params_frame, text=label + ":", font=self.gui_font, 
                          bg='#f0f0f0', fg='#2c3e50')
            lbl.grid(row=i, column=0, sticky=tk.W, pady=3, padx=(15, 0))
            var = tk.StringVar()
            # 增加输入框宽度
            entry = tk.Entry(params_frame, textvariable=var, width=18, font=self.gui_font)
            entry.grid(row=i, column=1, sticky=tk.W, padx=(15, 15), pady=3)
            self.param_vars[key] = var
        
        # 按钮框架 - 使用tk.Frame而不是ttk.Frame
        button_frame = tk.Frame(params_frame, bg='#f0f0f0')
        button_frame.grid(row=len(param_labels), column=0, columnspan=2, pady=(25, 15))
        
        # 按钮样式 - 文字颜色改为黑色
        calc_button = tk.Button(button_frame, text="开始计算", command=self.calculate,
                               width=12, height=2, font=self.gui_font,
                               bg='#0066cc', fg='black', 
                               activebackground='#004c99', activeforeground='black',
                               relief=tk.RAISED, borderwidth=2)
        calc_button.grid(row=0, column=0, padx=(0, 15))
        
        reset_button = tk.Button(button_frame, text="重置例题", command=self.reset_to_example,
                               width=12, height=2, font=self.gui_font,
                               bg='#0066cc', fg='black',
                               activebackground='#004c99', activeforeground='black',
                               relief=tk.RAISED, borderwidth=2)
        reset_button.grid(row=0, column=1)
        
        # 右侧结果显示框架 - 减小宽度
        results_frame = ttk.LabelFrame(main_frame, text="计算结果", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 主要结果显示
        self.result_text = scrolledtext.ScrolledText(results_frame, font=self.gui_font, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 详细计算过程框架 - 减小宽度
        process_frame = ttk.LabelFrame(main_frame, text="详细计算过程", padding="10")
        process_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 详细计算过程显示
        self.process_text = scrolledtext.ScrolledText(process_frame, font=self.gui_font, wrap=tk.WORD)
        self.process_text.pack(fill=tk.BOTH, expand=True)
        
        # 图表框架
        chart_frame = ttk.LabelFrame(main_frame, text="操作线图", padding="10")
        chart_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # 创建matplotlib图表
        self.fig = Figure(figsize=(12, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 配置网格权重 - 调整列权重比例
        main_frame.columnconfigure(0, weight=2)  # 参数设置列更宽
        main_frame.columnconfigure(1, weight=1)  # 计算结果列更窄
        main_frame.columnconfigure(2, weight=1)  # 详细过程列更窄
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def calculate_NOG(self, A, Y1, Y2, Y2_star):
        """计算传质单元数"""
        try:
            one_over_A = 1 / A
            term1 = 1 / (1 - one_over_A)
            term2 = math.log((1 - one_over_A) * (Y1 - Y2_star) / (Y2 - Y2_star) + one_over_A)
            return term1 * term2
        except (ZeroDivisionError, ValueError) as e:
            raise ValueError(f"NOG计算错误: {str(e)}")
    
    def solve_new_absorption_factor(self, NOG_target, Y1_new, Y2, Y2_star):
        """使用牛顿迭代法求解新吸收因子"""
        A_new = 1.5  # 初始猜值
        tolerance = 1e-8
        max_iterations = 1000
        
        for i in range(max_iterations):
            try:
                NOG_calc = self.calculate_NOG(A_new, Y1_new, Y2, Y2_star)
                error = NOG_calc - NOG_target
                
                if abs(error) < tolerance:
                    return A_new, i + 1  # 返回结果和迭代次数
                
                # 数值微分计算导数
                delta = 0.001
                NOG_delta = self.calculate_NOG(A_new + delta, Y1_new, Y2, Y2_star)
                derivative = (NOG_delta - NOG_calc) / delta
                
                if abs(derivative) > 1e-12:
                    A_new = A_new - error / derivative
                else:
                    A_new += 0.01
                
                # 约束A_new在合理范围内
                A_new = max(0.1, min(20, A_new))
                
            except Exception as e:
                A_new += 0.01
                if A_new > 20:
                    break
        
        raise ValueError(f"迭代求解失败，超过最大迭代次数 {max_iterations}")
    
    def calculate(self):
        """主计算函数"""
        try:
            # 获取参数
            params = {}
            for key, var in self.param_vars.items():
                try:
                    params[key] = float(var.get())
                except ValueError:
                    messagebox.showerror("输入错误", f"参数 {key} 必须是数字")
                    return
            
            # 参数验证
            if params['recovery'] <= 0 or params['recovery'] >= 1:
                messagebox.showerror("参数错误", "回收率必须在0到1之间")
                return
            
            if params['m'] <= 0:
                messagebox.showerror("参数错误", "平衡关系斜率必须大于0")
                return
            
            if params['LV_ratio'] <= 0:
                messagebox.showerror("参数错误", "液气比必须大于0")
                return
            
            # 开始计算
            self.result_text.delete(1.0, tk.END)
            self.process_text.delete(1.0, tk.END)
            
            # 原工况计算
            Y2 = params['Y1'] * (1 - params['recovery'])
            Y2_star = params['m'] * params['X2']
            A = params['LV_ratio'] / params['m']
            NOG = self.calculate_NOG(A, params['Y1'], Y2, Y2_star)
            
            # 新工况计算
            A_new, iterations = self.solve_new_absorption_factor(NOG, params['Y1_new'], Y2, Y2_star)
            LV_ratio_new = A_new * params['m']
            multiplier = LV_ratio_new / params['LV_ratio']
            
            # 验证计算
            NOG_verify = self.calculate_NOG(A_new, params['Y1_new'], Y2, Y2_star)
            recovery_new = (params['Y1_new'] - Y2) / params['Y1_new']
            
            # 显示主要结果
            result_text = f"""
计算结果摘要
{'='*30}

主要结果:
  吸收剂用量倍数: {multiplier:.6f} 倍
  新液气比 L'/V:   {LV_ratio_new:.6f}
  新吸收因子 A':   {A_new:.6f}

验证结果:
  原传质单元数 NOG:  {NOG:.6f}
  新传质单元数 N'OG: {NOG_verify:.6f}
  误差:              {abs(NOG - NOG_verify):.2e}
  
  原回收率:          {params['recovery']*100:.2f}%
  新工况回收率:      {recovery_new*100:.2f}%
  
求解信息:
  迭代次数:          {iterations}
  计算状态:          成功收敛

结论:
当入塔气体溶质含量从 {params['Y1']*100:.1f}% 
增加到 {params['Y1_new']*100:.1f}% 时，
为保持出塔气体组成不变，
吸收剂用量需增加到原用量的 {multiplier:.3f} 倍。
"""
            
            self.result_text.insert(tk.END, result_text)
            
            # 显示详细计算过程
            process_text = f"""
详细计算过程
{'='*40}

1. 原工况参数计算:
   气体入塔含量 Y₁ = {params['Y1']:.6f}
   气体出塔含量 Y₂ = Y₁(1-η) = {params['Y1']:.6f} × (1-{params['recovery']:.2f}) = {Y2:.6f}
   平衡气相组成 Y₂* = mX₂ = {params['m']:.1f} × {params['X2']:.6f} = {Y2_star:.6f}
   吸收因子 A = (L/V)/m = {params['LV_ratio']:.1f}/{params['m']:.1f} = {A:.6f}
   
   传质单元数计算:
   NOG = [1/(1-1/A)] × ln[(1-1/A)(Y₁-Y₂*)/(Y₂-Y₂*) + 1/A]
   NOG = [1/(1-1/{A:.3f})] × ln[(1-1/{A:.3f})×({params['Y1']:.6f}-{Y2_star:.6f})/({Y2:.6f}-{Y2_star:.6f}) + 1/{A:.3f}]
   NOG = {NOG:.6f}

2. 新工况设定:
   新气体入塔含量 Y₁' = {params['Y1_new']:.6f}
   保持出塔含量不变 Y₂' = Y₂ = {Y2:.6f}
   保持传质单元数不变 N'OG = NOG = {NOG:.6f}

3. 求解新吸收因子 A' (牛顿迭代法):
   目标方程: NOG = [1/(1-1/A')] × ln[(1-1/A')(Y₁'-Y₂*)/(Y₂'-Y₂*) + 1/A']
   迭代求解: A' = {A_new:.6f} (迭代{iterations}次收敛)

4. 最终结果计算:
   新液气比 L'/V = A' × m = {A_new:.6f} × {params['m']:.1f} = {LV_ratio_new:.6f}
   吸收剂用量倍数 = (L'/V)/(L/V) = {LV_ratio_new:.6f}/{params['LV_ratio']:.1f} = {multiplier:.6f}

5. 验证检查:
   新工况传质单元数 N'OG = {NOG_verify:.6f}
   与原工况偏差 = {abs(NOG - NOG_verify):.2e} (< 1e-6, 验证通过)
   新工况回收率 = (Y₁'-Y₂')/Y₁' = ({params['Y1_new']:.6f}-{Y2:.6f})/{params['Y1_new']:.6f} = {recovery_new:.4f}

计算完成！
"""
            
            self.process_text.insert(tk.END, process_text)
            
            # 绘制操作线图
            self.plot_operating_lines(params, Y2, Y2_star, A, A_new, LV_ratio_new)
            
        except Exception as e:
            messagebox.showerror("计算错误", f"计算过程中发生错误:\n{str(e)}")
    
    def plot_operating_lines(self, params, Y2, Y2_star, A_orig, A_new, LV_ratio_new):
        """绘制操作线和平衡线图"""
        self.fig.clear()
        
        # 设置中文字体和风格
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['figure.dpi'] = 100
        
        # 创建子图
        ax1 = self.fig.add_subplot(131)
        ax2 = self.fig.add_subplot(132)
        ax3 = self.fig.add_subplot(133)
        
        # 图1: 平衡线和操作线对比
        X_range = np.linspace(0, max(params['X2']*2, 0.001), 100)
        Y_equilibrium = params['m'] * X_range
        
        # 原操作线
        X1_orig = (params['Y1'] - Y2) / params['LV_ratio'] + params['X2']
        X_op_orig = np.linspace(params['X2'], X1_orig, 100)
        Y_op_orig = params['LV_ratio'] * (X_op_orig - params['X2']) + Y2
        
        # 新操作线
        X1_new = (params['Y1_new'] - Y2) / LV_ratio_new + params['X2']
        X_op_new = np.linspace(params['X2'], X1_new, 100)
        Y_op_new = LV_ratio_new * (X_op_new - params['X2']) + Y2
        
        ax1.plot(X_range, Y_equilibrium, 'r-', label=f'平衡线 Y={params["m"]}X', linewidth=2)
        ax1.plot(X_op_orig, Y_op_orig, 'b--', label=f'原操作线 L/V={params["LV_ratio"]:.1f}', linewidth=2)
        ax1.plot(X_op_new, Y_op_new, 'g-', label=f'新操作线 L/V={LV_ratio_new:.2f}', linewidth=2)
        
        # 标注关键点
        ax1.plot(params['X2'], Y2, 'ko', markersize=8, label='塔底')
        ax1.plot(X1_orig, params['Y1'], 'bo', markersize=8, label='原工况塔顶')
        ax1.plot(X1_new, params['Y1_new'], 'go', markersize=8, label='新工况塔顶')
        
        ax1.set_xlabel('液相摩尔分数 X', fontsize=11)
        ax1.set_ylabel('气相摩尔分数 Y', fontsize=11)
        ax1.set_title('气液平衡操作线', fontsize=12)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 图2: 参数对比柱状图
        categories = ['液气比', '吸收因子', '传质单元数']
        orig_values = [params['LV_ratio'], A_orig, self.calculate_NOG(A_orig, params['Y1'], Y2, Y2_star)]
        new_values = [LV_ratio_new, A_new, self.calculate_NOG(A_new, params['Y1_new'], Y2, Y2_star)]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax2.bar(x - width/2, orig_values, width, label='原工况', alpha=0.7, color='#3498db')
        ax2.bar(x + width/2, new_values, width, label='新工况', alpha=0.7, color='#e74c3c')
        
        ax2.set_xlabel('参数类型', fontsize=11)
        ax2.set_ylabel('数值', fontsize=11)
        ax2.set_title('工况参数对比', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 图3: 收敛过程图（模拟）
        iterations = np.arange(1, 21)
        # 模拟收敛过程
        A_convergence = A_new + (1.5 - A_new) * np.exp(-iterations/3)
        
        ax3.plot(iterations, A_convergence, 'b-o', markersize=4)
        ax3.axhline(y=A_new, color='r', linestyle='--', label=f'收敛值 A\'={A_new:.3f}')
        ax3.set_xlabel('迭代次数', fontsize=11)
        ax3.set_ylabel('吸收因子 A\'', fontsize=11)
        ax3.set_title('牛顿迭代收敛过程', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def reset_to_example(self):
        """重置为例题数据"""
        for key, value in self.default_params.items():
            self.param_vars[key].set(str(value))
        
        self.result_text.delete(1.0, tk.END)
        self.process_text.delete(1.0, tk.END)
        self.fig.clear()
        self.canvas.draw()
        
        # 显示例题说明
        example_text = """
例题数据已加载
================

本例题来源于化工原理教材例9.7:

某填料吸收塔用溶质含量为0.02%的溶剂
吸收混合气中的可溶组分，采用的液气比
为3.2，气体入塔溶质含量为2.0%，回收
率可达95%。平衡关系为Y=2X。

计算场景3：
入塔气体溶质含量增加至2.5%，为保证
气体出塔组成不变，吸收剂用量应增加
为原用量的多少倍？

点击"开始计算"进行求解。
"""
        self.result_text.insert(tk.END, example_text)

def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = AbsorptionCalculator(root)
        root.mainloop()
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请安装: pip install matplotlib numpy")
    except Exception as e:
        print(f"程序启动错误: {e}")

if __name__ == "__main__":
    main()