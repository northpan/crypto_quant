"""
可视化模块
生成回测结果的各种图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BacktestVisualizer:
    """
    回测可视化器
    
    生成图表:
    - 权益曲线
    - 回撤图
    - 月度收益热力图
    - 品种贡献分析
    - 交易分布
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ffbb78',
            'info': '#17becf'
        }
    
    def plot_equity_curve(
        self,
        equity_curve: Union[pd.Series, pd.DataFrame, List],
        benchmark: Optional[Union[pd.Series, List]] = None,
        title: str = "权益曲线",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制权益曲线
        
        Args:
            equity_curve: 权益曲线
            benchmark: 基准曲线
            title: 图表标题
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi,
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # 处理数据
        if isinstance(equity_curve, pd.DataFrame):
            if 'timestamp' in equity_curve.columns and 'equity' in equity_curve.columns:
                x = pd.to_datetime(equity_curve['timestamp'])
                y = equity_curve['equity']
            else:
                x = equity_curve.index
                y = equity_curve.iloc[:, 0]
        elif isinstance(equity_curve, pd.Series):
            x = equity_curve.index
            y = equity_curve.values
        else:
            x = range(len(equity_curve))
            y = equity_curve
        
        # 绘制权益曲线
        ax1.plot(x, y, label='Strategy', color=self.colors['primary'], linewidth=1.5)
        
        # 绘制基准
        if benchmark is not None:
            if isinstance(benchmark, pd.Series):
                bench_x = benchmark.index
                bench_y = benchmark.values
            else:
                bench_x = x[:len(benchmark)]
                bench_y = benchmark
            
            # 标准化基准
            if len(bench_y) > 0 and bench_y[0] != 0:
                bench_y = bench_y / bench_y[0] * y.iloc[0] if hasattr(y, 'iloc') else bench_y / bench_y[0] * y[0]
            
            ax1.plot(bench_x, bench_y, label='Benchmark', 
                    color=self.colors['secondary'], linewidth=1.5, alpha=0.7)
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('权益 ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 格式化y轴
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 绘制回撤
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = np.array(y)
        
        rolling_max = np.maximum.accumulate(y_values)
        drawdown = (y_values - rolling_max) / rolling_max * 100
        
        ax2.fill_between(x, drawdown, 0, color=self.colors['danger'], alpha=0.5)
        ax2.plot(x, drawdown, color=self.colors['danger'], linewidth=0.5)
        ax2.set_ylabel('回撤 (%)', fontsize=12)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 格式化x轴
        if isinstance(x, pd.DatetimeIndex) or (len(x) > 0 and isinstance(x[0], datetime)):
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_drawdown(
        self,
        equity_curve: Union[pd.Series, List],
        title: str = "回撤分析",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制回撤分析图"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize, dpi=self.dpi,
                                           gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # 处理数据
        if isinstance(equity_curve, pd.Series):
            x = equity_curve.index
            y = equity_curve.values
        else:
            x = range(len(equity_curve))
            y = np.array(equity_curve)
        
        # 权益曲线
        ax1.plot(x, y, color=self.colors['primary'], linewidth=1.5)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('权益 ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 回撤
        rolling_max = np.maximum.accumulate(y)
        drawdown = (y - rolling_max) / rolling_max * 100
        
        ax2.fill_between(x, drawdown, 0, color=self.colors['danger'], alpha=0.5)
        ax2.plot(x, drawdown, color=self.colors['danger'], linewidth=0.5)
        ax2.set_ylabel('回撤 (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 回撤持续期
        underwater = drawdown < 0
        durations = []
        current_duration = 0
        
        for is_under in underwater:
            if is_under:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        # 绘制回撤持续期分布
        if durations:
            ax3.hist(durations, bins=20, color=self.colors['warning'], alpha=0.7, edgecolor='black')
            ax3.set_xlabel('回撤持续期 (期)', fontsize=12)
            ax3.set_ylabel('频次', fontsize=12)
            ax3.set_title('回撤持续期分布', fontsize=12)
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_monthly_returns_heatmap(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        title: str = "月度收益热力图",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制月度收益热力图
        
        Args:
            returns: 收益率序列或DataFrame
            title: 图表标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(14, 8), dpi=self.dpi)
        
        # 处理数据
        if isinstance(returns, pd.Series):
            # 计算月度收益
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        else:
            monthly_returns = returns
        
        # 创建透视表
        if isinstance(monthly_returns, pd.Series):
            monthly_df = monthly_returns.to_frame('return')
            monthly_df['year'] = monthly_df.index.year
            monthly_df['month'] = monthly_df.index.month
            pivot = monthly_df.pivot(index='year', columns='month', values='return')
        else:
            pivot = monthly_returns
        
        # 重命名列
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = [month_names[i-1] for i in pivot.columns]
        
        # 绘制热力图
        sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=0, ax=ax, cbar_kws={'label': '收益率 (%)'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('月份', fontsize=12)
        ax.set_ylabel('年份', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_symbol_contribution(
        self,
        symbol_pnl: Dict[str, float],
        title: str = "品种贡献分析",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制品种贡献分析图
        
        Args:
            symbol_pnl: 品种盈亏字典 {symbol: pnl}
            title: 图表标题
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        # 排序
        sorted_pnl = dict(sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True))
        symbols = list(sorted_pnl.keys())
        pnls = list(sorted_pnl.values())
        
        # 颜色
        colors = [self.colors['success'] if p > 0 else self.colors['danger'] for p in pnls]
        
        # 柱状图
        ax1.barh(symbols, pnls, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('盈亏 ($)', fontsize=12)
        ax1.set_title('各品种盈亏贡献', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 饼图
        positive_pnl = sum(p for p in pnls if p > 0)
        negative_pnl = abs(sum(p for p in pnls if p < 0))
        
        if positive_pnl > 0 or negative_pnl > 0:
            sizes = [positive_pnl, negative_pnl]
            labels = [f'盈利\n${positive_pnl:,.2f}', f'亏损\n${negative_pnl:,.2f}']
            colors_pie = [self.colors['success'], self.colors['danger']]
            
            ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                   startangle=90, explode=(0.05, 0.05))
            ax2.set_title('盈亏占比', fontsize=12, fontweight='bold')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_trade_distribution(
        self,
        trades_df: pd.DataFrame,
        title: str = "交易分布分析",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制交易分布分析图
        
        Args:
            trades_df: 交易DataFrame
            title: 图表标题
            save_path: 保存路径
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = GridSpec(3, 2, figure=fig)
        
        # 盈亏分布
        ax1 = fig.add_subplot(gs[0, :])
        if 'pnl' in trades_df.columns:
            pnls = trades_df['pnl']
            ax1.hist(pnls, bins=30, color=self.colors['primary'], alpha=0.7, edgecolor='black')
            ax1.axvline(x=0, color='red', linestyle='--', linewidth=1)
            ax1.set_xlabel('盈亏 ($)', fontsize=11)
            ax1.set_ylabel('频次', fontsize=11)
            ax1.set_title('盈亏分布', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # 收益率分布
        ax2 = fig.add_subplot(gs[1, 0])
        if 'return_pct' in trades_df.columns:
            returns = trades_df['return_pct'] * 100
            ax2.hist(returns, bins=20, color=self.colors['secondary'], alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=1)
            ax2.set_xlabel('收益率 (%)', fontsize=11)
            ax2.set_ylabel('频次', fontsize=11)
            ax2.set_title('收益率分布', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 持仓时间分布
        ax3 = fig.add_subplot(gs[1, 1])
        if 'holding_hours' in trades_df.columns:
            holding = trades_df['holding_hours'].dropna()
            ax3.hist(holding, bins=20, color=self.colors['info'], alpha=0.7, edgecolor='black')
            ax3.set_xlabel('持仓时间 (小时)', fontsize=11)
            ax3.set_ylabel('频次', fontsize=11)
            ax3.set_title('持仓时间分布', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 按品种的交易次数
        ax4 = fig.add_subplot(gs[2, 0])
        if 'symbol' in trades_df.columns:
            symbol_counts = trades_df['symbol'].value_counts()
            ax4.bar(symbol_counts.index, symbol_counts.values, 
                   color=self.colors['warning'], alpha=0.7, edgecolor='black')
            ax4.set_xlabel('品种', fontsize=11)
            ax4.set_ylabel('交易次数', fontsize=11)
            ax4.set_title('各品种交易次数', fontsize=12, fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 盈亏散点图
        ax5 = fig.add_subplot(gs[2, 1])
        if 'return_pct' in trades_df.columns and 'holding_hours' in trades_df.columns:
            returns = trades_df['return_pct'] * 100
            holding = trades_df['holding_hours']
            colors = [self.colors['success'] if r > 0 else self.colors['danger'] for r in returns]
            ax5.scatter(holding, returns, c=colors, alpha=0.6, s=50)
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax5.set_xlabel('持仓时间 (小时)', fontsize=11)
            ax5.set_ylabel('收益率 (%)', fontsize=11)
            ax5.set_title('收益率 vs 持仓时间', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 30,
        title: str = "滚动指标",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制滚动指标图
        
        Args:
            returns: 收益率序列
            window: 滚动窗口
            title: 图表标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, dpi=self.dpi)
        
        # 滚动收益率
        rolling_return = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
        axes[0].plot(rolling_return.index, rolling_return * 100, 
                    color=self.colors['primary'], linewidth=1.5)
        axes[0].set_ylabel('滚动收益率 (%)', fontsize=11)
        axes[0].set_title(f'{window}期滚动指标', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        
        # 滚动波动率
        rolling_vol = returns.rolling(window).std() * np.sqrt(365) * 100
        axes[1].plot(rolling_vol.index, rolling_vol, 
                    color=self.colors['danger'], linewidth=1.5)
        axes[1].set_ylabel('滚动波动率 (%)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # 滚动夏普
        rolling_mean = returns.rolling(window).mean() * 365
        rolling_std = returns.rolling(window).std() * np.sqrt(365)
        rolling_sharpe = (rolling_mean - 0.02) / rolling_std
        axes[2].plot(rolling_sharpe.index, rolling_sharpe, 
                    color=self.colors['success'], linewidth=1.5)
        axes[2].set_ylabel('滚动夏普比率', fontsize=11)
        axes[2].set_xlabel('时间', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_comprehensive_report(
        self,
        equity_curve: Union[pd.Series, pd.DataFrame],
        returns: pd.Series,
        trades_df: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
        title: str = "回测综合报告",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制综合报告
        
        Args:
            equity_curve: 权益曲线
            returns: 收益率序列
            trades_df: 交易DataFrame
            benchmark: 基准收益率
            title: 图表标题
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. 权益曲线
        ax1 = fig.add_subplot(gs[0, :2])
        if isinstance(equity_curve, pd.DataFrame):
            if 'timestamp' in equity_curve.columns:
                x = pd.to_datetime(equity_curve['timestamp'])
                y = equity_curve['equity']
            else:
                x = equity_curve.index
                y = equity_curve.iloc[:, 0]
        else:
            x = equity_curve.index
            y = equity_curve.values
        
        ax1.plot(x, y, color=self.colors['primary'], linewidth=1.5, label='Strategy')
        if benchmark is not None:
            bench_y = benchmark.values if hasattr(benchmark, 'values') else benchmark
            if len(bench_y) > 0 and bench_y[0] != 0:
                bench_y = bench_y / bench_y[0] * y.iloc[0] if hasattr(y, 'iloc') else bench_y / bench_y[0] * y[0]
            bench_x = x[:len(bench_y)]
            ax1.plot(bench_x, bench_y, color=self.colors['secondary'], 
                    linewidth=1.5, alpha=0.7, label='Benchmark')
        ax1.set_title('权益曲线', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. 回撤
        ax2 = fig.add_subplot(gs[0, 2])
        y_values = y.values if hasattr(y, 'values') else np.array(y)
        rolling_max = np.maximum.accumulate(y_values)
        drawdown = (y_values - rolling_max) / rolling_max * 100
        ax2.fill_between(x, drawdown, 0, color=self.colors['danger'], alpha=0.5)
        ax2.set_title('回撤', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 月度收益
        ax3 = fig.add_subplot(gs[1, 0])
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        colors = [self.colors['success'] if r > 0 else self.colors['danger'] for r in monthly_returns]
        ax3.bar(range(len(monthly_returns)), monthly_returns * 100, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('月度收益', fontsize=12, fontweight='bold')
        ax3.set_ylabel('收益率 (%)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 收益率分布
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(returns * 100, bins=30, color=self.colors['primary'], alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax4.set_title('收益率分布', fontsize=12, fontweight='bold')
        ax4.set_xlabel('收益率 (%)')
        ax4.grid(True, alpha=0.3)
        
        # 5. 滚动夏普
        ax5 = fig.add_subplot(gs[1, 2])
        window = 30
        rolling_mean = returns.rolling(window).mean() * 365
        rolling_std = returns.rolling(window).std() * np.sqrt(365)
        rolling_sharpe = (rolling_mean - 0.02) / rolling_std
        ax5.plot(rolling_sharpe.index, rolling_sharpe, color=self.colors['success'], linewidth=1.5)
        ax5.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        ax5.set_title('滚动夏普 (30期)', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. 交易盈亏分布
        if trades_df is not None and 'pnl' in trades_df.columns:
            ax6 = fig.add_subplot(gs[2, 0])
            pnls = trades_df['pnl']
            ax6.hist(pnls, bins=20, color=self.colors['info'], alpha=0.7, edgecolor='black')
            ax6.axvline(x=0, color='red', linestyle='--', linewidth=1)
            ax6.set_title('交易盈亏分布', fontsize=12, fontweight='bold')
            ax6.set_xlabel('盈亏 ($)')
            ax6.grid(True, alpha=0.3)
        
        # 7. 品种贡献
        if trades_df is not None and 'symbol' in trades_df.columns and 'pnl' in trades_df.columns:
            ax7 = fig.add_subplot(gs[2, 1])
            symbol_pnl = trades_df.groupby('symbol')['pnl'].sum()
            colors = [self.colors['success'] if p > 0 else self.colors['danger'] for p in symbol_pnl]
            ax7.barh(symbol_pnl.index, symbol_pnl.values, color=colors, alpha=0.7)
            ax7.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax7.set_title('品种贡献', fontsize=12, fontweight='bold')
            ax7.set_xlabel('盈亏 ($)')
            ax7.grid(True, alpha=0.3, axis='x')
        
        # 8. 累计收益
        ax8 = fig.add_subplot(gs[2, 2])
        cumulative = (1 + returns).cumprod()
        ax8.plot(cumulative.index, cumulative, color=self.colors['primary'], linewidth=1.5)
        ax8.set_title('累计收益', fontsize=12, fontweight='bold')
        ax8.set_ylabel('倍数')
        ax8.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def save_all_plots(
        self,
        output_dir: str,
        equity_curve: Union[pd.Series, pd.DataFrame],
        returns: pd.Series,
        trades_df: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
        symbol_pnl: Optional[Dict[str, float]] = None
    ):
        """保存所有图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 权益曲线
        self.plot_equity_curve(
            equity_curve, benchmark,
            save_path=f"{output_dir}/equity_curve.png"
        )
        plt.close()
        
        # 回撤分析
        self.plot_drawdown(
            equity_curve,
            save_path=f"{output_dir}/drawdown.png"
        )
        plt.close()
        
        # 月度收益热力图
        self.plot_monthly_returns_heatmap(
            returns,
            save_path=f"{output_dir}/monthly_returns.png"
        )
        plt.close()
        
        # 交易分布
        if trades_df is not None:
            self.plot_trade_distribution(
                trades_df,
                save_path=f"{output_dir}/trade_distribution.png"
            )
            plt.close()
        
        # 品种贡献
        if symbol_pnl is not None:
            self.plot_symbol_contribution(
                symbol_pnl,
                save_path=f"{output_dir}/symbol_contribution.png"
            )
            plt.close()
        
        # 滚动指标
        self.plot_rolling_metrics(
            returns,
            save_path=f"{output_dir}/rolling_metrics.png"
        )
        plt.close()
        
        # 综合报告
        self.plot_comprehensive_report(
            equity_curve, returns, trades_df, benchmark,
            save_path=f"{output_dir}/comprehensive_report.png"
        )
        plt.close()
        
        print(f"所有图表已保存到: {output_dir}")


if __name__ == "__main__":
    # 测试可视化
    print("可视化模块测试")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    n_days = 365
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    
    # 收益率
    returns = np.random.normal(0.001, 0.02, n_days)
    returns = pd.Series(returns, index=dates)
    
    # 权益曲线
    equity = 100000 * (1 + returns).cumprod()
    
    # 基准
    benchmark_returns = np.random.normal(0.0005, 0.015, n_days)
    benchmark = 100000 * (1 + benchmark_returns).cumprod()
    
    # 交易数据
    trades_data = []
    for i in range(50):
        trades_data.append({
            'trade_id': f'TRD_{i:03d}',
            'symbol': np.random.choice(['BTCUSDT', 'ETHUSDT', 'SOLUSDT']),
            'pnl': np.random.normal(100, 500),
            'return_pct': np.random.normal(0.01, 0.05),
            'holding_hours': np.random.randint(1, 48)
        })
    trades_df = pd.DataFrame(trades_data)
    
    # 品种盈亏
    symbol_pnl = {
        'BTCUSDT': 5000,
        'ETHUSDT': 3000,
        'SOLUSDT': -1000
    }
    
    # 创建可视化器
    visualizer = BacktestVisualizer(figsize=(14, 10), dpi=100)
    
    # 测试各个图表
    print("生成权益曲线...")
    fig = visualizer.plot_equity_curve(equity, benchmark)
    plt.show()
    
    print("生成回撤分析...")
    fig = visualizer.plot_drawdown(equity)
    plt.show()
    
    print("生成月度收益热力图...")
    fig = visualizer.plot_monthly_returns_heatmap(returns)
    plt.show()
    
    print("生成品种贡献分析...")
    fig = visualizer.plot_symbol_contribution(symbol_pnl)
    plt.show()
    
    print("生成交易分布分析...")
    fig = visualizer.plot_trade_distribution(trades_df)
    plt.show()
    
    print("生成滚动指标...")
    fig = visualizer.plot_rolling_metrics(returns, window=30)
    plt.show()
    
    print("生成综合报告...")
    fig = visualizer.plot_comprehensive_report(equity, returns, trades_df, benchmark)
    plt.show()
    
    print("\n测试完成!")
