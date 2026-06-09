@echo off
:: 强制当前控制台使用 UTF-8 编码
chcp 65001 >nul

echo ====================================
echo 启动 SuperBizAgent 服务 (Conda 环境版)
echo ====================================
echo.

:: 核心：直接使用你当前环境的 python 变量，不再寻找任何 .venv 文件夹
set PYTHON_CMD=python

echo [1/4] 启动 Milvus 向量数据库...
docker ps --format "{{.Names}}" | findstr "milvus-standalone" >nul 2>&1
if not errorlevel 1 (
    echo [信息] Milvus 容器已在运行
) else (
    docker compose -f vector-database.yml up -d
    if errorlevel 1 (
        echo [错误] Docker 启动失败，请确保 Docker Desktop 已启动
        pause
        exit /b 1
    )
    echo [信息] 等待 Milvus 启动（10秒）...
    timeout /t 10 /nobreak >nul
)
echo [成功] Milvus 数据库就绪
echo.

echo [2/4] 启动 CLS MCP 服务...
start "CLS MCP Server" /min %PYTHON_CMD% mcp_servers/cls_server.py
timeout /t 2 /nobreak >nul
echo [成功] CLS MCP 服务已启动
echo.

echo [3/4] 启动 Monitor MCP 服务...
start "Monitor MCP Server" /min %PYTHON_CMD% mcp_servers/monitor_server.py
timeout /t 2 /nobreak >nul
echo [成功] Monitor MCP 服务已启动
echo.

echo [4/4] 启动 FastAPI 主服务...
start "SuperBizAgent API" %PYTHON_CMD% -m uvicorn app.main:app --host 0.0.0.0 --port 9900
echo [信息] 等待服务初始化（15秒）...
timeout /t 15 /nobreak >nul
echo.

echo [信息] 检查服务状态并自动同步知识库文档...
curl -s http://localhost:9900/health >nul 2>&1
if errorlevel 1 (
    echo [警告] 服务启动稍慢，知识库未自动上传，请稍后手动上传。
) else (
    for %%f in (aiops-docs\*.md) do (
        echo   正在索引: %%~nxf
        curl -s -X POST http://localhost:9900/api/upload -F "file=@%%f" >nul 2>&1
    )
    echo [成功] 知识库文档同步完成！
)

echo.
echo ====================================
echo 所有服务已顺畅启动！
echo Web 界面: http://localhost:9900
echo API 文档: http://localhost:9900/docs
echo ====================================
pause