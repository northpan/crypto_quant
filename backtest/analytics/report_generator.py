"""
报告生成器
生成完整的回测报告，包括文本报告和HTML报告
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import json
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


class ReportGenerator:
    """
    回测报告生成器
    
    生成:
    - 文本报告
    - HTML报告
    - JSON数据
    - Markdown报告
    """
    
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_data: Dict[str, Any] = {}
    
    def generate_text_report(
        self,
        performance_metrics: Dict,
        trade_metrics: Optional[Dict] = None,
        overfitting_results: Optional[Dict] = None,
        strategy_info: Optional[Dict] = None
    ) -> str:
        """
        生成文本报告
        
        Args:
            performance_metrics: 绩效指标
            trade_metrics: 交易指标
            overfitting_results: 过拟合测试结果
            strategy_info: 策略信息
        
        Returns:
            文本报告
        """
        lines = []
        
        # 标题
        lines.extend([
            "=" * 80,
            " " * 30 + "回测报告",
            "=" * 80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ])
        
        # 策略信息
        if strategy_info:
            lines.extend([
                "策略信息",
                "-" * 80,
                f"策略名称: {strategy_info.get('name', 'Unknown')}",
                f"策略描述: {strategy_info.get('description', 'N/A')}",
                f"交易品种: {strategy_info.get('symbols', 'N/A')}",
                f"时间周期: {strategy_info.get('timeframe', 'N/A')}",
                f"回测区间: {strategy_info.get('start_date', 'N/A')} ~ {strategy_info.get('end_date', 'N/A')}",
                ""
            ])
        
        # 绩效指标
        if performance_metrics:
            lines.extend([
                "绩效指标",
                "-" * 80,
                f"初始资金:          ${performance_metrics.get('initial_capital', 0):>15,.2f}",
                f"最终权益:          ${performance_metrics.get('final_value', 0):>15,.2f}",
                f"总盈亏:            ${performance_metrics.get('total_pnl', 0):>15,.2f}",
                f"总收益率:          {performance_metrics.get('total_return', 0)*100:>15.2f}%",
                f"年化收益率:        {performance_metrics.get('annualized_return', 0)*100:>15.2f}%",
                f"年化波动率:        {performance_metrics.get('annualized_volatility', 0)*100:>15.2f}%",
                f"最大回撤:          {performance_metrics.get('max_drawdown', 0)*100:>15.2f}%",
                f"夏普比率:          {performance_metrics.get('sharpe_ratio', 0):>15.2f}",
                f"索提诺比率:        {performance_metrics.get('sortino_ratio', 0):>15.2f}",
                f"Calmar比率:        {performance_metrics.get('calmar_ratio', 0):>15.2f}",
                f"Alpha:             {performance_metrics.get('alpha', 0)*100:>15.2f}%",
                f"Beta:              {performance_metrics.get('beta', 0):>15.2f}",
                ""
            ])
        
        # 交易指标
        if trade_metrics:
            lines.extend([
                "交易统计",
                "-" * 80,
                f"总交易次数:        {trade_metrics.get('total_trades', 0):>15}",
                f"盈利交易:          {trade_metrics.get('winning_trades', 0):>15}",
                f"亏损交易:          {trade_metrics.get('losing_trades', 0):>15}",
                f"胜率:              {trade_metrics.get('win_rate', 0)*100:>15.2f}%",
                f"盈亏比:            {trade_metrics.get('profit_factor', 0):>15.2f}",
                f"平均盈利:          ${trade_metrics.get('avg_profit', 0):>15.2f}",
                f"平均亏损:          ${trade_metrics.get('avg_loss', 0):>15.2f}",
                f"最大盈利:          ${trade_metrics.get('max_profit', 0):>15.2f}",
                f"最大亏损:          ${trade_metrics.get('max_loss', 0):>15.2f}",
                f"期望值:            ${trade_metrics.get('expectancy', 0):>15.2f}",
                f"SQN:               {trade_metrics.get('sqn', 0):>15.2f}",
                ""
            ])
        
        # 过拟合检测
        if overfitting_results:
            lines.extend([
                "过拟合检测",
                "-" * 80
            ])
            
            for test_name, result in overfitting_results.items():
                lines.extend([
                    f"{test_name}:",
                    f"  是否过拟合:      {'是' if result.get('is_overfitted', False) else '否'}",
                    f"  概率:            {result.get('probability', 0)*100:.2f}%",
                    f"  建议:            {result.get('recommendation', 'N/A')}",
                    ""
                ])
        
        lines.extend([
            "=" * 80,
            "报告结束",
            "=" * 80
        ])
        
        return "\n".join(lines)
    
    def generate_html_report(
        self,
        performance_metrics: Dict,
        trade_metrics: Optional[Dict] = None,
        overfitting_results: Optional[Dict] = None,
        strategy_info: Optional[Dict] = None,
        chart_paths: Optional[List[str]] = None
    ) -> str:
        """
        生成HTML报告
        
        Args:
            performance_metrics: 绩效指标
            trade_metrics: 交易指标
            overfitting_results: 过拟合测试结果
            strategy_info: 策略信息
            chart_paths: 图表路径列表
        
        Returns:
            HTML报告
        """
        # 嵌入图表
        chart_html = ""
        if chart_paths:
            for path in chart_paths:
                if Path(path).exists():
                    with open(path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                    chart_html += f'<img src="data:image/png;base64,{img_data}" style="max-width:100%;margin:20px 0;"><br>'
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>回测报告 - {strategy_info.get('name', 'Strategy') if strategy_info else 'Strategy'}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .positive {{
            color: #28a745;
        }}
        .negative {{
            color: #dc3545;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .alert {{
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .alert-warning {{
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            color: #856404;
        }}
        .alert-success {{
            background-color: #d4edda;
            border: 1px solid #28a745;
            color: #155724;
        }}
        .alert-danger {{
            background-color: #f8d7da;
            border: 1px solid #dc3545;
            color: #721c24;
        }}
        h1, h2 {{
            margin-top: 0;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>回测报告</h1>
        <p>策略: {strategy_info.get('name', 'Unknown') if strategy_info else 'Unknown'}</p>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>绩效概览</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if performance_metrics.get('total_return', 0) > 0 else 'negative'}">
                    {performance_metrics.get('total_return', 0)*100:.2f}%
                </div>
                <div class="metric-label">总收益率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {performance_metrics.get('sharpe_ratio', 0):.2f}
                </div>
                <div class="metric-label">夏普比率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">
                    {performance_metrics.get('max_drawdown', 0)*100:.2f}%
                </div>
                <div class="metric-label">最大回撤</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {performance_metrics.get('annualized_volatility', 0)*100:.2f}%
                </div>
                <div class="metric-label">年化波动率</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>详细绩效指标</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>数值</th>
                <th>指标</th>
                <th>数值</th>
            </tr>
            <tr>
                <td>初始资金</td>
                <td>${performance_metrics.get('initial_capital', 0):,.2f}</td>
                <td>最终权益</td>
                <td>${performance_metrics.get('final_value', 0):,.2f}</td>
            </tr>
            <tr>
                <td>总盈亏</td>
                <td class="{'positive' if performance_metrics.get('total_pnl', 0) > 0 else 'negative'}">
                    ${performance_metrics.get('total_pnl', 0):,.2f}
                </td>
                <td>年化收益率</td>
                <td>{performance_metrics.get('annualized_return', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>索提诺比率</td>
                <td>{performance_metrics.get('sortino_ratio', 0):.2f}</td>
                <td>Calmar比率</td>
                <td>{performance_metrics.get('calmar_ratio', 0):.2f}</td>
            </tr>
            <tr>
                <td>Alpha</td>
                <td>{performance_metrics.get('alpha', 0)*100:.2f}%</td>
                <td>Beta</td>
                <td>{performance_metrics.get('beta', 0):.2f}</td>
            </tr>
        </table>
    </div>
"""
        
        # 交易统计：卡片 + 详细表
        if trade_metrics:
            win_rate_pct = trade_metrics.get('win_rate', 0) * 100
            avg_holding = trade_metrics.get('avg_holding_period', 0) or 0
            if hasattr(avg_holding, 'total_seconds'):
                avg_holding_str = f"{avg_holding.total_seconds() / 3600:.1f}h"
            elif isinstance(avg_holding, (int, float)):
                avg_holding_str = f"{avg_holding:.1f}h" if avg_holding else "—"
            else:
                avg_holding_str = str(avg_holding) if avg_holding else "—"
            html += f"""
    <div class="section">
        <h2>交易统计</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{trade_metrics.get('total_trades', 0)}</div>
                <div class="metric-label">总交易次数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if win_rate_pct >= 50 else 'negative'}">{win_rate_pct:.1f}%</div>
                <div class="metric-label">胜率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{trade_metrics.get('profit_factor', 0):.2f}</div>
                <div class="metric-label">盈亏比</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_holding_str}</div>
                <div class="metric-label">平均持仓时长</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{trade_metrics.get('max_consecutive_losses', 0)}</div>
                <div class="metric-label">最大连续亏损次数</div>
            </div>
        </div>
        <table>
            <tr>
                <th>指标</th>
                <th>数值</th>
                <th>指标</th>
                <th>数值</th>
            </tr>
            <tr>
                <td>总交易次数</td>
                <td>{trade_metrics.get('total_trades', 0)}</td>
                <td>胜率</td>
                <td>{win_rate_pct:.2f}%</td>
            </tr>
            <tr>
                <td>盈利交易</td>
                <td>{trade_metrics.get('winning_trades', 0)}</td>
                <td>亏损交易</td>
                <td>{trade_metrics.get('losing_trades', 0)}</td>
            </tr>
            <tr>
                <td>盈亏比</td>
                <td>{trade_metrics.get('profit_factor', 0):.2f}</td>
                <td>期望值</td>
                <td>${trade_metrics.get('expectancy', 0):.2f}</td>
            </tr>
            <tr>
                <td>平均持仓时长</td>
                <td>{avg_holding_str}</td>
                <td>最大连续亏损</td>
                <td>{trade_metrics.get('max_consecutive_losses', 0)}</td>
            </tr>
            <tr>
                <td>最大盈利</td>
                <td class="positive">${trade_metrics.get('max_profit', 0):.2f}</td>
                <td>最大亏损</td>
                <td class="negative">${trade_metrics.get('max_loss', 0):.2f}</td>
            </tr>
        </table>
    </div>
"""
        
        # 过拟合检测
        if overfitting_results:
            html += """
    <div class="section">
        <h2>过拟合检测</h2>
"""
            for test_name, result in overfitting_results.items():
                alert_class = "alert-danger" if result.get('is_overfitted', False) else "alert-success"
                html += f"""
        <div class="alert {alert_class}">
            <strong>{test_name}</strong><br>
            是否过拟合: {'是' if result.get('is_overfitted', False) else '否'} | 
            概率: {result.get('probability', 0)*100:.2f}%<br>
            {result.get('recommendation', '')}
        </div>
"""
            html += "    </div>"
        
        # 图表
        if chart_html:
            html += f"""
    <div class="section">
        <h2>图表分析</h2>
        {chart_html}
    </div>
"""
        
        html += """
    <div class="footer">
        <p>Generated by CryptoQuant Backtest Engine</p>
    </div>
</body>
</html>
"""
        
        return html
    
    def generate_json_report(
        self,
        performance_metrics: Dict,
        trade_metrics: Optional[Dict] = None,
        overfitting_results: Optional[Dict] = None,
        strategy_info: Optional[Dict] = None
    ) -> str:
        """
        生成JSON报告
        
        Returns:
            JSON字符串
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'strategy_info': strategy_info or {},
            'performance': performance_metrics,
            'trades': trade_metrics or {},
            'overfitting': overfitting_results or {}
        }
        
        return json.dumps(report, indent=2, default=str)
    
    def generate_markdown_report(
        self,
        performance_metrics: Dict,
        trade_metrics: Optional[Dict] = None,
        overfitting_results: Optional[Dict] = None,
        strategy_info: Optional[Dict] = None
    ) -> str:
        """
        生成Markdown报告
        
        Returns:
            Markdown字符串
        """
        md = f"""# 回测报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        if strategy_info:
            md += f"""## 策略信息

| 项目 | 值 |
|------|-----|
| 策略名称 | {strategy_info.get('name', 'Unknown')} |
| 策略描述 | {strategy_info.get('description', 'N/A')} |
| 交易品种 | {strategy_info.get('symbols', 'N/A')} |
| 时间周期 | {strategy_info.get('timeframe', 'N/A')} |
| 回测区间 | {strategy_info.get('start_date', 'N/A')} ~ {strategy_info.get('end_date', 'N/A')} |

"""
        
        md += f"""## 绩效概览

| 指标 | 数值 |
|------|------|
| 总收益率 | {performance_metrics.get('total_return', 0)*100:.2f}% |
| 年化收益率 | {performance_metrics.get('annualized_return', 0)*100:.2f}% |
| 夏普比率 | {performance_metrics.get('sharpe_ratio', 0):.2f} |
| 最大回撤 | {performance_metrics.get('max_drawdown', 0)*100:.2f}% |
| 年化波动率 | {performance_metrics.get('annualized_volatility', 0)*100:.2f}% |

## 详细绩效指标

| 指标 | 数值 | 指标 | 数值 |
|------|------|------|------|
| 初始资金 | ${performance_metrics.get('initial_capital', 0):,.2f} | 最终权益 | ${performance_metrics.get('final_value', 0):,.2f} |
| 总盈亏 | ${performance_metrics.get('total_pnl', 0):,.2f} | 年化收益率 | {performance_metrics.get('annualized_return', 0)*100:.2f}% |
| 索提诺比率 | {performance_metrics.get('sortino_ratio', 0):.2f} | Calmar比率 | {performance_metrics.get('calmar_ratio', 0):.2f} |
| Alpha | {performance_metrics.get('alpha', 0)*100:.2f}% | Beta | {performance_metrics.get('beta', 0):.2f} |

"""
        
        if trade_metrics:
            md += f"""## 交易统计

| 指标 | 数值 | 指标 | 数值 |
|------|------|------|------|
| 总交易次数 | {trade_metrics.get('total_trades', 0)} | 胜率 | {trade_metrics.get('win_rate', 0)*100:.2f}% |
| 盈利交易 | {trade_metrics.get('winning_trades', 0)} | 亏损交易 | {trade_metrics.get('losing_trades', 0)} |
| 盈亏比 | {trade_metrics.get('profit_factor', 0):.2f} | 期望值 | ${trade_metrics.get('expectancy', 0):.2f} |
| 最大盈利 | ${trade_metrics.get('max_profit', 0):.2f} | 最大亏损 | ${trade_metrics.get('max_loss', 0):.2f} |

"""
        
        if overfitting_results:
            md += """## 过拟合检测

| 测试 | 是否过拟合 | 概率 | 建议 |
|------|-----------|------|------|
"""
            for test_name, result in overfitting_results.items():
                md += f"| {test_name} | {'是' if result.get('is_overfitted', False) else '否'} | {result.get('probability', 0)*100:.2f}% | {result.get('recommendation', 'N/A')} |\n"
            md += "\n"
        
        md += """---

*Generated by CryptoQuant Backtest Engine*
"""
        
        return md
    
    def save_all_reports(
        self,
        performance_metrics: Dict,
        trade_metrics: Optional[Dict] = None,
        overfitting_results: Optional[Dict] = None,
        strategy_info: Optional[Dict] = None,
        chart_paths: Optional[List[str]] = None,
        prefix: str = "backtest_report"
    ) -> Dict[str, str]:
        """
        保存所有格式的报告
        
        Returns:
            保存的文件路径字典
        """
        saved_files = {}
        
        # 文本报告
        text_report = self.generate_text_report(
            performance_metrics, trade_metrics, overfitting_results, strategy_info
        )
        text_path = self.output_dir / f"{prefix}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        saved_files['text'] = str(text_path)
        
        # HTML报告
        html_report = self.generate_html_report(
            performance_metrics, trade_metrics, overfitting_results, strategy_info, chart_paths
        )
        html_path = self.output_dir / f"{prefix}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        saved_files['html'] = str(html_path)
        
        # JSON报告
        json_report = self.generate_json_report(
            performance_metrics, trade_metrics, overfitting_results, strategy_info
        )
        json_path = self.output_dir / f"{prefix}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_report)
        saved_files['json'] = str(json_path)
        
        # Markdown报告
        md_report = self.generate_markdown_report(
            performance_metrics, trade_metrics, overfitting_results, strategy_info
        )
        md_path = self.output_dir / f"{prefix}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        saved_files['markdown'] = str(md_path)
        
        print(f"报告已保存到: {self.output_dir}")
        return saved_files


if __name__ == "__main__":
    # 测试报告生成器
    print("报告生成器测试")
    print("=" * 60)
    
    # 示例数据
    performance_metrics = {
        'initial_capital': 100000,
        'final_value': 125000,
        'total_pnl': 25000,
        'total_return': 0.25,
        'annualized_return': 0.30,
        'annualized_volatility': 0.20,
        'max_drawdown': -0.15,
        'sharpe_ratio': 1.40,
        'sortino_ratio': 1.80,
        'calmar_ratio': 2.00,
        'alpha': 0.05,
        'beta': 0.80
    }
    
    trade_metrics = {
        'total_trades': 150,
        'winning_trades': 90,
        'losing_trades': 60,
        'win_rate': 0.60,
        'profit_factor': 1.50,
        'avg_profit': 350,
        'avg_loss': -200,
        'max_profit': 1500,
        'max_loss': -800,
        'expectancy': 110,
        'sqn': 2.50
    }
    
    overfitting_results = {
        'out_of_sample': {
            'is_overfitted': False,
            'probability': 0.20,
            'recommendation': '样本外表现良好'
        },
        'monte_carlo': {
            'is_overfitted': False,
            'probability': 0.15,
            'recommendation': '蒙特卡洛测试通过'
        }
    }
    
    strategy_info = {
        'name': 'Dual MA Crossover',
        'description': '双均线交叉策略',
        'symbols': 'BTCUSDT, ETHUSDT',
        'timeframe': '1H',
        'start_date': '2023-01-01',
        'end_date': '2024-01-01'
    }
    
    # 创建报告生成器
    generator = ReportGenerator(output_dir="./test_reports")
    
    # 生成所有报告
    saved_files = generator.save_all_reports(
        performance_metrics=performance_metrics,
        trade_metrics=trade_metrics,
        overfitting_results=overfitting_results,
        strategy_info=strategy_info
    )
    
    print("\n生成的报告:")
    for format_type, path in saved_files.items():
        print(f"  {format_type}: {path}")
    
    # 打印文本报告
    print("\n" + "=" * 60)
    print("文本报告预览:")
    print("=" * 60)
    print(generator.generate_text_report(
        performance_metrics, trade_metrics, overfitting_results, strategy_info
    ))
