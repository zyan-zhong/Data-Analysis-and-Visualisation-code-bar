#基于 Matplotlib 绘图函数
#画图初始化
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

#绘制基础线图 (x 轴的数据（列表或数组）,y 轴的数据（列表或数组）)
plt.plot(x, y)

    x = [1, 2, 3, 4]
    y = [10, 15, 13, 17]
    plt.plot(x, y)
    plt.show() # 记得用show()来显示图表

#绘制散点图 (x 轴的数据（列表或数组）,y 轴的数据（列表或数组）)
plt.scatter(x, y)

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    plt.scatter(x, y)
    plt.show()

#绘制柱状图 (x: 柱子的标签或位置（例如类别的名称）,height: 每个柱子的高度（即对应的数值）
plt.bar(x, height)

    x = ['A', 'B', 'C', 'D']
    height = [5, 7, 3, 4] 
    plt.bar(x, height)
    plt.show()

#绘制直方图
plt.hist(data)

    # 生成一些随机数据：1000个符合正态分布的数
    data = np.random.randn(1000)
    plt.hist(data)
    plt.show()

#绘制饼图 (labels: 每个扇区的标签,sizes: 每个扇区的大小)
plt.pie(sizes, labels=labels)

    labels = ['A', 'B', 'C', 'D']
    sizes = [15, 30, 45, 10]
    plt.pie(sizes, labels=labels)
    plt.show()

#绘制箱线图 (data: 数据集，可以是列表、数组或DataFrame)
plt.boxplot(data)       

    data = [np.random.randn(100) for _ in range(4)]  # 生成4组随机数据
    plt.boxplot(data)
    plt.show()
#绘制热力图 (data: 2D数组或DataFrame)
plt.imshow(data, cmap='hot', interpolation='nearest')
    data = np.random.rand(10, 10)  # 生成10x10的随机数据
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar()  # 显示颜色条
    plt.show()

#绘制子图 (nrows: 子图的行数,ncols: 子图的列数)
fig, axs = plt.subplots(nrows, ncols)

    fig, axs = plt.subplots(2, 2)  # 创建2x2的子图
    axs[0, 0].plot(x, y)  # 在第一个子图绘制线图
    axs[0, 1].scatter(x, y)  # 在第二个子图绘制散点图
    axs[1, 0].bar(x, height)  # 在第三个子图绘制柱状图
    axs[1, 1].hist(data)  # 在第四个子图绘制直方图
    plt.show()

#图表美化与信息添加
# 添加标题
plt.title('Title of the Chart')

    plt.plot(x, y)
    plt.title('My First Plot') # 添加标题
    plt.show()

# 添加X轴标签
plt.xlabel('X-axis Label')

    plt.plot(x, y)
    plt.xlabel('X-axis') # 添加X轴标签
    plt.show()

# 添加Y轴标签
plt.ylabel('Y-axis')

    plt.plot(x, y)
    plt.ylabel('Y-axis') # 添加Y轴标签  
    plt.show()

# 显示图例
plt.legend()

    plt.plot(x, y, label='Line 1')
    plt.plot(x2, y2, label='Line 2')    
    plt.legend() # 显示图例
    plt.show()  

# 显示网格
plt.grid(True)

    plt.plot(x, y)
    plt.grid(True) # 显示网格
    plt.show()

# 显示图表
plt.show()

#完整的综合基础用法代码示例
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. 创建一些示例数据
    x = np.linspace(0, 10, 100) # 生成0到10之间100个均匀间隔的数
    y1 = np.sin(x) # y1是x的正弦值
    y2 = np.cos(x) # y2是x的余弦值

    # 2. 创建画布和绘图
    plt.figure(figsize=(10, 6)) # 设置图的大小（可选）
    plt.plot(x, y1, label='sin(x)') # 画正弦曲线
    plt.plot(x, y2, label='cos(x)') # 画余弦曲线

    # 3. 添加标签和标题
    plt.title('Sine and Cosine Waves')
    plt.xlabel('x axis')
    plt.ylabel('y axis')

    # 4. 添加图例和网格
    plt.legend()
    plt.grid(True)

    # 5. 显示图表
    plt.show()


#基于 Plotly 绘图函数
import plotly.express as px

# 绘制折线图
fig = px.line(data_frame, x='x_column', y='y_column', title='Line Plot')
fig.show()

    import plotly.express as px
    df = px.data.gapminder().query("country=='Canada'") # 示例数据
    fig = px.line(df, x='year', y='lifeExp', title='Life expectancy in Canada')
    fig.show()

# 绘制散点图
fig = px.scatter(data_frame, x='x_column', y='y_column', title='Scatter Plot')
fig.show()

    df = px.data.iris() # 著名的鸢尾花数据集
    fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')
    fig.show()
    # 这个图会用不同颜色区分不同种类的花，一目了然。

# 绘制柱状图
fig = px.bar(data_frame, x='x_column', y='y_column', title='Bar Chart')
fig.show()

    df = px.data.medals_long() # 示例数据：奖牌数
    fig = px.bar(df, x='nation', y='count', color='medal')
    fig.show()
    # 这个会生成一个堆叠或分组的柱状图，显示每个国家的金、银、铜牌数量。

# 绘制直方图
fig = px.histogram(data_frame, x='value_column', title='Histogram')
fig.show()

    import numpy as np
    np.random.seed(0)
    data = np.random.randn(1000) # 生成1000个随机数
    import pandas as pd
    df = pd.DataFrame({'values': data}) # 将数据转换为DataFrame
    fig = px.histogram(df, x='values')
    fig.show()

# 绘制饼图
fig = px.pie(data_frame, names='category_column', values='value_column', title='Pie Chart')
fig.show()

    df = px.data.tips() # 示例数据：小费数据
    fig = px.pie(df, names='day', values='total_bill', title='Total Bill by Day')
    fig.show()

# 绘制箱线图
fig = px.box(data_frame, y='value_column', title='Box Plot')
fig.show()

    df = px.data.tips() # 示例数据：小费数据
    fig = px.box(df, x='day', y='total_bill', title='Total Bill by Day')
    fig.show()

#绘制热力图
fig = px.imshow(data_matrix, title='Heatmap')
fig.show()

    import numpy as np
    data = np.random.rand(10, 10) # 生成10x10的随机数据矩阵
    fig = px.imshow(data, title='Random Heatmap')
    fig.show()

#图表美化与信息添加 (Figure 对象的方法)
#创建完 fig 对象后，我们用以下方法来修饰它。这些方法可以链式调用（如 fig.update_layout(...).update_xaxes(...)

fig.update_layout(title='Chart Title', xaxis_title='X Axis', yaxis_title='Y Axis', legend_title='Legend')
fig.update_xaxes(showgrid=True, zeroline=True, gridwidth=1, gridcolor='LightGray')
fig.update_yaxes(showgrid=True, zeroline=True, gridwidth=1, gridcolor='LightGray')
fig.show()

# 添加总标题
fig.update_layout(title='Title')

    import plotly.express as px
    df = px.data.gapminder().query("country=='Canada'") # 示例数据
    fig = px.line(df, x='year', y='lifeExp', title='Life expectancy in Canada')
    fig.update_layout(title='Life Expectancy in Canada Over Years')
    fig.show()

# 设置X轴标题
fig.update_layout(xaxis_title='X-axis Label')

    fig.update_layout(xaxis_title='Year')
    fig.show()

# 设置Y轴标题
fig.update_layout(yaxis_title='Y-axis Label')

    fig.update_layout(yaxis_title='Life Expectancy')
    fig.show()

# 设置图例标签
fig.update_layout(legend_title='Legend Title')

    fig.update_layout(legend_title='Legend')
    fig.show()
# 控制图例显示
fig.update_layout(legend=dict(x=0, y=1)) # 位置 (0,1) 是左上角
fig.show()

# 显示图表
fig.show()


#基于 Plotly Express 绘图函数
# 标准导入
import plotly.express as px
import pandas as pd
import numpy as np

#绘制折线图
fig = px.line(data_frame, x='x_column', y='y_column', title='Line Plot')
fig.show()

    import plotly.express as px
    df = px.data.gapminder().query("country=='Canada'") # 示例数据
    fig = px.line(df, x='year', y='lifeExp', title='Life expectancy in Canada')
    fig.show()

#绘制散点图
fig = px.scatter(data_frame, x='x_column', y='y_column', title='Scatter Plot')
fig.show()

    df = px.data.iris() # 著名的鸢尾花数据集
    fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')
    fig.show()
    # 这个图会用不同颜色区分不同种类的花，一目了然。

#绘制柱状图
fig = px.bar(data_frame, x='x_column', y='y_column', title='Bar Chart')
fig.show()

    df = px.data.medals_long() # 示例数据：奖牌数
    fig = px.bar(df, x='nation', y='count', color='medal')
    fig.show()

    # 这个会生成一个堆叠或分组的柱状图，显示每个国家的金、银、铜牌数量。

#绘制直方图
fig = px.histogram(data_frame, x='value_column', title='Histogram')
fig.show()

    import numpy as np
    np.random.seed(0)
    data = np.random.randn(1000) # 生成1000个随机数
    import pandas as pd
    df = pd.DataFrame({'values': data}) # 将数据转换为DataFrame
    fig = px.histogram(df, x='values')
    fig.show()

#绘制饼图
fig = px.pie(data_frame, names='category_column', values='value_column', title
='Pie Chart')
fig.show()

    df = px.data.tips() # 示例数据：小费数据
    fig = px.pie(df, names='day', values='total_bill', title='Total Bill by Day')
    fig.show()

#绘制箱线图
fig = px.box(data_frame, y='value_column', title='Box Plot')
fig.show()

    df = px.data.tips() # 示例数据：小费数据
    fig = px.box(df, x='day', y='total_bill', title='Total Bill by Day')
    fig.show()

#绘制热力图
fig = px.imshow(data_matrix, title='Heatmap')
fig.show()

    import numpy as np
    data = np.random.rand(10, 10) # 生成10x10的随机数据矩阵
    fig = px.imshow(data, title='Random Heatmap')
    fig.show()

#图表定制化
#添加总标题
fig.update_layout(title='Title')
fig.show()

    fig = px.line(data_frame=df, x='年份', y='销售额')
    fig.update_layout(title='年度销售额趋势分析')
    fig.show()

#设置X轴标题
fig.update_layout(xaxis_title='X-axis Label')

    fig = px.line(data_frame=df, x='年份', y='销售额')
    fig.update_xaxes(title_text='年份')
    fig.show()

#设置Y轴标题
fig.update_layout(yaxis_title='Y-axis Label')

    fig = px.line(data_frame=df, x='年份', y='销售额')
    fig.update_yaxes(title_text='销售额 (万元)')
    fig.show()

#设置图例标签
fig.update_traces(name='图例标签')

    # 创建包含多个系列的数据
    df = pd.DataFrame({
        '年份': [2010, 2011, 2012, 2013, 2014, 2015],
        '产品A': [100, 120, 90, 150, 200, 180],
        '产品B': [80, 90, 100, 110, 130, 150]
    })

    # 将数据从宽格式转换为长格式
    df_long = df.melt(id_vars='年份', var_name='产品', value_name='销售额')

    # 创建图表
    fig = px.line(data_frame=df_long, x='年份', y='销售额', color='产品')
    fig.update_traces(name='产品系列')  # 设置图例标题
    fig.show()

#控制图例显示
fig.update_layout(showlegend=True)

    fig = px.line(data_frame=df_long, x='年份', y='销售额', color='产品')
    fig.update_layout(showlegend=True)  # 确保图例显示
    fig.show()

#Matplotlib库
#标准导入
import matplotlib.pyplot as plt
import numpy as np

    #折线图 (plt.plot)
    plt.plot(x_data, y_data, format_string, label='图例标签')

    # 创建示例数据
    x = np.linspace(0, 10, 100)  # 0到10之间的100个等间距点
    y = np.sin(x)  # 计算每个点的正弦值

    # 创建折线图
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.plot(x, y, 'b-', linewidth=2, label='正弦曲线')  # 蓝色实线，线宽为2

    # 添加标题和标签
    plt.title('正弦函数曲线')
    plt.xlabel('X轴')
    plt.ylabel('Y轴')

    # 添加图例和网格
    plt.legend()
    plt.grid(True)

    # 显示图形
    plt.show()

#散点图 (plt.scatter)
plt.scatter(x_data, y_data, s=size, c=color, marker='o', label='图例标签')

    # 创建示例数据
    x = np.random.rand(50) * 10  # 50个0到10之间的随机数
    y = np.random.rand(50) * 10  # 50个0到10之间的随机数    
    sizes = np.random.rand(50) * 100  # 点的大小
    colors = np.random.rand(50)  # 点的颜色 
    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=sizes, c=colors, alpha=0.6, cmap='viridis', label='随机点')  # 使用viridis颜色映射
    # 添加标题和标签
    plt.title('随机散点图')
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    # 添加图例和网格
    plt.legend()    
    plt.grid(True)

    # 显示图形
    plt.show()  

#柱状图 (plt.bar)
plt.bar(x_categories, height_values, width=bar_width, label='图例标签')

    # 计算每个种类的数量
    species_count = [np.sum(iris_target == i) for i in range(3)]

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(iris_target_names,  # x轴类别
                species_count,      # 柱子的高度
                color=['lightblue', 'lightgreen', 'lightcoral'],  # 柱子颜色
                alpha=0.7,          # 透明度
                width=0.6)          # 柱子宽度

    # 添加标题和标签
    plt.title('鸢尾花各类别数量')
    plt.xlabel('鸢尾花种类')
    plt.ylabel('数量')

    # 在每个柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')

    # 显示图形
    plt.show()

#直方图 (plt.hist)
plt.hist(data, bins=bin_count, alpha=transparency, label='图例标签')

    # 创建直方图
    plt.figure(figsize=(10, 6))

    # 绘制花瓣长度的直方图
    plt.hist(iris_data[:, 2],  # 花瓣长度数据
            bins=20,          # 分为20个区间
            color='skyblue',  # 颜色
            alpha=0.7,        # 透明度
            edgecolor='black') # 边界颜色

    # 添加标题和标签
    plt.title('鸢尾花花瓣长度分布')
    plt.xlabel('花瓣长度 (cm)')
    plt.ylabel('频数')

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 显示图形
    plt.show()

#图表定制化
#添加标题
plt.title('标题文本', fontsize=字体大小, fontweight='字重')

#添加X轴标签 (plt.xlabel)
plt.xlabel('X轴标签文本', fontsize=字体大小)

#添加Y轴标签 (plt.ylabel)
plt.ylabel('Y轴标签文本', fontsize=字体大小)

#显示图例 (plt.legend)
plt.legend(loc='位置代码', fontsize=字体大小)

#显示网格 (plt.grid)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=透明度)

#显示图表 (plt.show)
plt.show()

#保存图形
plt.savefig('文件名.png', dpi=分辨率, bbox_inches='tight')

#设置图形大小
plt.figure(figsize=(宽度, 高度))
plt.figure(figsize=(10, 6)) # 宽10英寸，高6英寸

#设置线条样式
plt.plot(x, y, linestyle='--', color='r', marker='o', markersize=5)
# linestyle: 线条样式，如 '-' (实线), '--' (虚线), '-.' (点划线), ':' (点线)
# color: 线条颜色，如 'r' (红色), 'g' (绿色), 'b' (蓝色), 'k' (黑色) 等
# marker: 数据点标记样式，如 'o' (圆圈), 's' (方块), '^' (三角形) 等
# markersize: 标记大小

#设置坐标轴范围
plt.xlim(最小值, 最大值)
plt.ylim(最小值, 最大值)
plt.xlim(0, 10) # 设置X轴范围为0到10
plt.ylim(-1, 1) # 设置Y轴范围为-1到1
#设置刻度 (plt.xticks, plt.yticks)
plt.xticks(ticks=刻度列表, labels=标签列表, rotation=旋转角度)
plt.yticks(ticks=刻度列表, labels=标签列表, rotation=旋转角度)
plt.xticks(ticks=[0, 2, 4, 6, 8, 10], labels=['零', '二', '四', '六', '八', '十'], rotation=45)
# ticks: 刻度位置列表   
# labels: 刻度标签列表
# rotation: 标签旋转角度

#设置字体 (plt.rcParams)
plt.rcParams['font.family'] = '字体名称'
plt.rcParams['font.size'] = 字体大小
plt.rcParams['font.weight'] = '字重'
plt.rcParams['font.family'] = 'Arial' # 设置字体为Arial
plt.rcParams['font.size'] = 12        # 设置字体大小为12
plt.rcParams['font.weight'] = 'bold'  # 设置字体为粗体

#添加文本注释
plt.text(x位置, y位置, '文本内容', fontsize=字体大小, color='颜色')
plt.text(5, 0, '峰值', fontsize=12, color='red
') # 在坐标(5,0)处添加文本'峰值'    

