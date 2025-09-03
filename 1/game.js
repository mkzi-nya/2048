// Version: 2 (Alpha-Beta 取代所有旧 AI / worker.js，并基于确定性种子+步数搜索)
// 本代码实现：使用确定性搜索 + Alpha-Beta 剪枝 + 迭代加深，自动游玩任何棋盘大小
// 每当得到一段确定主线（剪枝结果的主变着，PV），就以 0.1s/步 自动播放

// ============ 全局变量 ============
let side = 4;
let base = 2;
let tiles = [];
let tileID = 0;
let currentScore = 0;
let bestScore = 0;
let isGameStarted = false;
let globalStep = 0;  // 步数计数器（用于确定性生成新砖）
const SWIPE_THRESHOLD = 10;

// ===== 辅助函数：格式化数值 =====
function formatNumber(n) {
  const units = ["", "k", "m", "g", "t", "p", "e", "z", "y", "r", "q"];
  let unitIndex = 0;

  while (n >= 10240 && unitIndex < units.length - 1) {
    n = Math.floor(n / 1024);
    unitIndex++;
  }

  return n + units[unitIndex];
}

// ============ 页面加载 ============
window.addEventListener("DOMContentLoaded", () => {
  const bestScoreKey = "bestScore_" + side + "_" + base;
  if(localStorage.getItem(bestScoreKey)) {
    bestScore = parseInt(localStorage.getItem(bestScoreKey));
  }
  document.getElementById("bestScore").innerText = formatNumber(bestScore);
  
  document.getElementById("spanSide").innerText = side;
  document.getElementById("spanBase").innerText = base;

  createGrid(side);

  document.getElementById("btnSideMinus").addEventListener("click", () => changeSide(-1));
  document.getElementById("btnSidePlus").addEventListener("click", () => changeSide(+1));
  document.getElementById("btnBaseMinus").addEventListener("click", () => changeBase(-1));
  document.getElementById("btnBasePlus").addEventListener("click", () => changeBase(+1));

  document.getElementById("btnApply").addEventListener("click", applySettings);
  document.getElementById("btnRestart").addEventListener("click", restartGame);
  document.getElementById("btnDarkMode").addEventListener("click", toggleDarkMode);
  
  document.getElementById("btnExport").addEventListener("click", exportSave);
  document.getElementById("btnImport").addEventListener("click", () => {
    document.getElementById("fileInput").click();
  });
  document.getElementById("fileInput").addEventListener("change", importSave);

  const gridContainer = document.getElementById("gridContainer");
  gridContainer.addEventListener("touchmove", (e)=> e.preventDefault(), {passive:false});

  if (localStorage.getItem("darkMode") === "true") {
    document.body.classList.add("dark-mode");
  }
});

// ============ 创建背景格子 ============
function createGrid(s) {
  const gridContainer = document.getElementById("gridContainer");
  gridContainer.innerHTML = "";
  gridContainer.style.gridTemplateRows = `repeat(${s}, 1fr)`;
  gridContainer.style.gridTemplateColumns = `repeat(${s}, 1fr)`;
  for(let i = 0; i < s; i++){
    for(let j = 0; j < s; j++){
      const cell = document.createElement("div");
      cell.classList.add("grid-cell");
      gridContainer.appendChild(cell);
    }
  }
}

// ============ 改变棋盘边长 ============
function changeSide(delta) {
  if(isGameStarted) return;
  let newVal = side + delta;
  if(newVal < 2) newVal = 2;
  if(newVal > 10) newVal = 10;
  side = newVal;
  document.getElementById("spanSide").innerText = side;
  const bestScoreKey = "bestScore_" + side + "_" + base;
  let stored = localStorage.getItem(bestScoreKey);
  document.getElementById("bestScore").innerText = stored ? formatNumber(parseInt(stored)) : "0";
  createGrid(side);
}

// ============ 改变基数 ============
function changeBase(delta) {
  if(isGameStarted) return;
  let newBase = base + delta;
  if(newBase < 2) newBase = 2;
  if(newBase > 10) newBase = 10;
  base = newBase;
  document.getElementById("spanBase").innerText = base;
  const bestScoreKey = "bestScore_" + side + "_" + base;
  let stored = localStorage.getItem(bestScoreKey);
  document.getElementById("bestScore").innerText = stored ? formatNumber(parseInt(stored)) : "0";
}

// ============ 点击“开始游戏” ============
function applySettings() {
  if(isGameStarted) return;
  isGameStarted = true;
  document.getElementById("btnSideMinus").disabled = true;
  document.getElementById("btnSidePlus").disabled = true;
  document.getElementById("btnBaseMinus").disabled = true;
  document.getElementById("btnBasePlus").disabled = true;
  document.getElementById("btnApply").disabled = true;

  const bestScoreKey = "bestScore_" + side + "_" + base;
  if(localStorage.getItem(bestScoreKey)) {
    bestScore = parseInt(localStorage.getItem(bestScoreKey));
  } else {
    bestScore = 0;
  }
  document.getElementById("bestScore").innerText = formatNumber(bestScore);
  setupGame();
}

// ============ 初始化/重置游戏 ============
function setupGame() {
  currentScore = 0;
  document.getElementById("currentScore").innerText = "0";
  tiles = [];
  tileID = 0;
  
  // 生成新的游戏种子
  globalSeed = Math.floor(Math.random() * 1000000);
  console.log("游戏种子:", globalSeed);
  
  createGrid(side);
  spawnRandomTile();
  spawnRandomTile();
  initInputEvents();
  renderAllTiles(true);
}

// ============ 重新开始 ============
function restartGame() {
  isGameStarted = false;
  document.getElementById("btnSideMinus").disabled = false;
  document.getElementById("btnSidePlus").disabled = false;
  document.getElementById("btnBaseMinus").disabled = false;
  document.getElementById("btnBasePlus").disabled = false;
  document.getElementById("btnApply").disabled = false;
  tiles = [];
  tileID = 0;
  currentScore = 0;
  globalSeed = null; // 重置种子
  globalStep = 0;
  document.getElementById("currentScore").innerText = "0";
  const oldTiles = document.querySelectorAll(".tile");
  oldTiles.forEach(t => t.remove());
  createGrid(side);
}

// ============ 暗色模式切换 ============
function toggleDarkMode() {
  document.body.classList.toggle("dark-mode");
  localStorage.setItem("darkMode", document.body.classList.contains("dark-mode"));
}

// ============ 分数更新 ============
function updateScore(add) {
  currentScore += add;
  document.getElementById("currentScore").innerText = formatNumber(currentScore);
  if(currentScore > bestScore) {
    bestScore = currentScore;
    document.getElementById("bestScore").innerText = formatNumber(bestScore);
    const bestScoreKey = "bestScore_" + side + "_" + base;
    localStorage.setItem(bestScoreKey, bestScore);
  }
}

// ============ 输入事件（键盘+触摸） ============
let inputInited = false;
function initInputEvents() {
  if(inputInited) return;
  inputInited = true;
  document.addEventListener("keydown", (e) => {
    if(!isGameStarted) return;
    let moved = false;
    switch(e.key) {
      case "ArrowUp":    moved = moveUp();    break;
      case "ArrowDown":  moved = moveDown();  break;
      case "ArrowLeft":  moved = moveLeft();  break;
      case "ArrowRight": moved = moveRight(); break;
      default: return;
    }
    if(moved) {
      renderAllTiles();
      setTimeout(() => {
        finalizePositions();
        spawnRandomTile();
        renderAllTiles();
        checkGameOver();
      }, 200);
    }
  });
  const gridContainer = document.getElementById("gridContainer");
  let startX = 0, startY = 0;
  gridContainer.addEventListener("touchstart", (e) => {
    if(!isGameStarted) return;
    if(e.touches.length > 0) {
      startX = e.touches[0].clientX;
      startY = e.touches[0].clientY;
    }
  }, {passive:false});
  gridContainer.addEventListener("touchend", (e) => {
    if(!isGameStarted) return;
    if(e.changedTouches.length > 0) {
      let endX = e.changedTouches[0].clientX;
      let endY = e.changedTouches[0].clientY;
      let dx = endX - startX;
      let dy = endY - startY;
      if(Math.abs(dx) < SWIPE_THRESHOLD && Math.abs(dy) < SWIPE_THRESHOLD) return;
      let moved = false;
      if(Math.abs(dx) > Math.abs(dy)) {
        moved = dx > 0 ? moveRight() : moveLeft();
      } else {
        moved = dy > 0 ? moveDown() : moveUp();
      }
      if(moved) {
        renderAllTiles();
        setTimeout(() => {
          finalizePositions();
          spawnRandomTile();
          renderAllTiles();
          checkGameOver();
        }, 200);
      }
    }
  }, {passive:false});
}

// 更新所有方块的状态
function finalizePositions() {
  tiles.forEach(t => {
    t.oldRow = t.row;
    t.oldCol = t.col;
    t.merged = false;
  });
}

// ============ 移动与合并逻辑 ============
function moveLeft(){
  tiles.forEach(t => { t.oldRow = t.row; t.oldCol = t.col; t.merged = false; });
  let moved = false;
  for (let r = 0; r < side; r++) {
    let rowTiles = tiles.filter(t => t.row === r);
    rowTiles.sort((a, b) => a.col - b.col);
    if(compressLine(rowTiles, "left")) moved = true;
  }
  return moved;
}

function moveRight(){
  tiles.forEach(t => { t.oldRow = t.row; t.oldCol = t.col; t.merged = false; });
  let moved = false;
  for (let r = 0; r < side; r++) {
    let rowTiles = tiles.filter(t => t.row === r);
    rowTiles.sort((a, b) => b.col - a.col);
    if(compressLine(rowTiles, "right")) moved = true;
  }
  return moved;
}

function moveUp(){
  tiles.forEach(t => { t.oldRow = t.row; t.oldCol = t.col; t.merged = false; });
  let moved = false;
  for (let c = 0; c < side; c++) {
    let colTiles = tiles.filter(t => t.col === c);
    colTiles.sort((a, b) => a.row - b.row);
    if(compressLine(colTiles, "up")) moved = true;
  }
  return moved;
}

function moveDown(){
  tiles.forEach(t => { t.oldRow = t.row; t.oldCol = t.col; t.merged = false; });
  let moved = false;
  for (let c = 0; c < side; c++) {
    let colTiles = tiles.filter(t => t.col === c);
    colTiles.sort((a, b) => b.row - a.row);
    if(compressLine(colTiles, "down")) moved = true;
  }
  return moved;
}

/**
 * 合并一行或一列的方块  
 * 当两个相邻且数值相同（即等级 n 相同）的方块合并后，新 tile 的等级 n 增加 1，
 * 数值更新为：value = base × 2^(n–1)
 */
function compressLine(lineTiles, direction) {
  if(lineTiles.length === 0) return false;
  let changed = false;
  let newLine = [];
  let skip = false;
  for (let i = 0; i < lineTiles.length; i++) {
    if(skip) { skip = false; continue; }
    if(i < lineTiles.length - 1 && lineTiles[i].value === lineTiles[i+1].value) {
      let targetTile = lineTiles[i];
      let mergingTile = lineTiles[i+1];
      targetTile.n = targetTile.n + 1;
      targetTile.value = base * Math.pow(2, targetTile.n - 1);
      updateScore(targetTile.value);
      targetTile.merged = true;
      targetTile.mergedFrom = { row: mergingTile.row, col: mergingTile.col };
      targetTile.oldRow = mergingTile.row;
      targetTile.oldCol = mergingTile.col;
      newLine.push(targetTile);
      tiles = tiles.filter(x => x.id !== mergingTile.id);
      skip = true;
      changed = true;
    } else {
      newLine.push(lineTiles[i]);
    }
  }
  if(direction === "left") {
    for (let i = 0; i < newLine.length; i++) {
      let tile = newLine[i];
      let oldCol = tile.col;
      tile.col = i;
      if(tile.col !== oldCol) changed = true;
    }
  } else if(direction === "right") {
    for (let i = 0; i < newLine.length; i++) {
      let tile = newLine[i];
      let oldCol = tile.col;
      tile.col = side - 1 - i;
      if(tile.col !== oldCol) changed = true;
    }
  } else if(direction === "up") {
    for (let i = 0; i < newLine.length; i++) {
      let tile = newLine[i];
      let oldRow = tile.row;
      tile.row = i;
      if(tile.row !== oldRow) changed = true;
    }
  } else if(direction === "down") {
    for (let i = 0; i < newLine.length; i++) {
      let tile = newLine[i];
      let oldRow = tile.row;
      tile.row = side - 1 - i;
      if(tile.row !== oldRow) changed = true;
    }
  }
  return changed;
}

// ============ 生成随机方块（确定性） ============
let globalSeed = null;

// 确定性伪随机数生成器 (Mulberry32算法)
function createDeterministicRandom(seed) {
    return function() {
        seed |= 0;
        seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// 生成确定性的随机方块信息（返回第n个空位和数值）
function generateTileInfo(seed, step, boardSize) {
    const totalCells = boardSize * boardSize;
    const random = createDeterministicRandom(seed + step * 12345);
    const emptySlotIndex = Math.floor(random() * totalCells);
    const value = random() < 0.8 ? 2 : 4;
    return {
        emptySlotIndex: emptySlotIndex,
        value: value
    };
}

// 在 spawnRandomTile 中使用：
function spawnRandomTile() {
    if (globalSeed === null) {
        console.error("错误: 游戏种子未初始化");
        return;
    }
    
    const tileInfo = generateTileInfo(globalSeed, globalStep, side);
    
    // 获取所有空位
    const emptyCells = [];
    for (let r = 0; r < side; r++) {
        for (let c = 0; c < side; c++) {
            if (!tiles.some(t => t.row === r && t.col === c)) {
                emptyCells.push({ r, c });
            }
        }
    }
    
    if (emptyCells.length === 0) return;
    const selectedIndex = tileInfo.emptySlotIndex % emptyCells.length;
    const selectedCell = emptyCells[selectedIndex];
    
    console.log(`步数 ${globalStep}: 在第${selectedIndex}个空位(${selectedCell.r},${selectedCell.c})生成${tileInfo.value}`);
    
    const level = tileInfo.value === 2 ? 1 : 2;
    tiles.push({
        id: ++tileID,
        value: tileInfo.value,
        n: level,
        row: selectedCell.r,
        col: selectedCell.c,
        oldRow: selectedCell.r,
        oldCol: selectedCell.c,
        merged: false,
        spawned: true
    });
    globalStep++;  // 每次生成方块后步数+1
}

// ============ 渲染方块 ============
function renderAllTiles(skipAnimation = false) {
  const container = document.getElementById("gridContainer");
  const bgCells = container.querySelectorAll(".grid-cell");
  const oldDOM = container.querySelectorAll(".tile");
  oldDOM.forEach(d => d.remove());
  tiles.forEach(t => {
    const div = document.createElement("div");
    div.classList.add("tile", "tile-n" + t.n);
    if(t.spawned) {
      div.classList.add("tile-pop");
      div.style.transitionDuration = "0.1s";
    } else if(t.merged) {
      div.classList.add("tile-merged");
      div.style.transitionDuration = "0.2s";
    } else {
      div.style.transitionDuration = "0.2s";
    }
    div.innerText = formatNumber(t.value);
    let oldRect = getCellRect(bgCells, t.oldRow, t.oldCol);
    div.style.width = oldRect.width + "px";
    div.style.height = oldRect.height + "px";
    div.style.left = oldRect.left + "px";
    div.style.top = oldRect.top + "px";
    div.style.fontSize = Math.floor(oldRect.width * 0.4) + "px";
    container.appendChild(div);
  });
  if(skipAnimation) {
    tiles.forEach(t => { t.oldRow = t.row; t.oldCol = t.col; });
    tiles.forEach((t, idx) => {
      let tileDiv = container.querySelectorAll(".tile")[idx];
      let newRect = getCellRect(bgCells, t.row, t.col);
      tileDiv.style.left = newRect.left + "px";
      tileDiv.style.top = newRect.top + "px";
      tileDiv.style.width = newRect.width + "px";
      tileDiv.style.height = newRect.height + "px";
      tileDiv.style.fontSize = Math.floor(newRect.width * 0.4) + "px";
    });
    return;
  }
  requestAnimationFrame(() => {
    tiles.forEach((t, idx) => {
      let tileDiv = container.querySelectorAll(".tile")[idx];
      let newRect = getCellRect(bgCells, t.row, t.col);
      tileDiv.style.left = newRect.left + "px";
      tileDiv.style.top = newRect.top + "px";
      tileDiv.style.width = newRect.width + "px";
      tileDiv.style.height = newRect.height + "px";
      tileDiv.style.fontSize = Math.floor(newRect.width * 0.4) + "px";
    });
  });
  tiles.forEach(t => { t.spawned = false; });
}

// 辅助：获取格子位置信息
function getCellRect(bgCells, row, col) {
  const index = row * side + col;
  const cellElem = bgCells[index];
  const container = document.getElementById("gridContainer");
  let cellRect = cellElem.getBoundingClientRect();
  let containerRect = container.getBoundingClientRect();
  return {
    left: cellRect.left - containerRect.left,
    top: cellRect.top - containerRect.top,
    width: cellRect.width,
    height: cellRect.height
  };
}

// ============ 检查游戏结束 ============
function checkGameOver() {
  if(tiles.length < side * side) return;
  for(let i = 0; i < tiles.length; i++){
    for(let j = i + 1; j < tiles.length; j++){
      let a = tiles[i], b = tiles[j];
      if(a.value === b.value) {
        if(a.row === b.row && Math.abs(a.col - b.col) === 1) return;
        if(a.col === b.col && Math.abs(a.row - b.row) === 1) return;
      }
    }
  }
  alert("游戏结束！最终分数：" + formatNumber(currentScore));
}

// ============ 存档导出 ============
function exportSave() {
  let bestScores = {};
  for(let key in localStorage) {
    if(key.startsWith("bestScore_")) {
      bestScores[key] = localStorage.getItem(key);
    }
  }
  let saveData = {
    version: 1,
    bestScores: bestScores,
    currentGame: {
      isGameStarted: isGameStarted,
      currentScore: currentScore,
      side: side,
      base: base,
      tileID: tileID,
      tiles: tiles,
      darkMode: document.body.classList.contains("dark-mode"),
      seed: globalSeed,
      step: globalStep
    }
  };
  let dataStr = JSON.stringify(saveData);
  let blob = new Blob([dataStr], {type: "application/json"});
  let url = URL.createObjectURL(blob);
  let a = document.createElement("a");
  a.href = url;
  a.download = "2048_save.json";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ============ 存档导入 ============
function importSave(e) {
  let file = e.target.files[0];
  if (!file) return;
  let reader = new FileReader();
  reader.onload = function(evt) {
    try {
      let data = JSON.parse(evt.target.result);
      if(data.version !== 1) {
        alert("存档版本不匹配！");
        return;
      }
      for(let key in data.bestScores) {
        localStorage.setItem(key, data.bestScores[key]);
      }
      let game = data.currentGame;
      isGameStarted = game.isGameStarted;
      currentScore = game.currentScore;
      side = game.side;
      base = game.base;
      tileID = game.tileID;
      tiles = game.tiles;
      
      if (game.seed) {
        globalSeed = game.seed;
        console.log("恢复游戏种子:", globalSeed);
      } else {
        globalSeed = Math.floor(Math.random() * 1000000);
        console.log("生成新游戏种子:", globalSeed);
      }
      if (game.step !== undefined) {
        globalStep = game.step;
        console.log("恢复游戏步数:", globalStep);
      } else {
        globalStep = tiles.length;
        console.log("从方块数量推算步数:", globalStep);
      }
      if(game.darkMode) {
        document.body.classList.add("dark-mode");
      } else {
        document.body.classList.remove("dark-mode");
      }
      document.getElementById("spanSide").innerText = side;
      document.getElementById("spanBase").innerText = base;
      const bestScoreKey = "bestScore_" + side + "_" + base;
      let stored = localStorage.getItem(bestScoreKey);
      document.getElementById("bestScore").innerText = stored ? formatNumber(parseInt(stored)) : "0";
      document.getElementById("currentScore").innerText = formatNumber(currentScore);
      createGrid(side);
      renderAllTiles(true);
      if(!inputInited) {
        initInputEvents();
      }
    } catch(err) {
      alert("读取存档失败！");
    }
  };
  reader.readAsText(file);
}

/* =========================
   以下为 新 AI：确定性搜索 + Alpha-Beta 剪枝
   关键点：
   1) 使用 simulate* 在“本地拷贝状态”下进行移动与合并（捕获得分），不污染全局
   2) 按你的 generateTileInfo(seed, step) 规则，在搜索树里“确定性地”生成新砖
   3) Alpha-Beta + 迭代加深 + 置换表 + 简单启发式（单调性/角权/空格）
   4) 每次得到一段主变着（PV），以 0.1s/步 自动播放
========================= */

// —— 工具：把当前 tiles 深拷贝为可模拟状态（仅保留必要字段）
function cloneState(srcTiles) {
  return JSON.parse(JSON.stringify(srcTiles));
}

// —— 工具：在给定 state 下，统计空位（行优先）
function listEmptyCells(state) {
  const occ = new Set(state.map(t => `${t.row},${t.col}`));
  const res = [];
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      const key = `${r},${c}`;
      if (!occ.has(key)) res.push({r, c});
    }
  }
  return res;
}

// —— 在不改动全局的情况下模拟一次移动（已复用你原来的拦截 updateScore 的逻辑）
function simulateMoveOnState(moveFunc, state) {
  let backupTiles = tiles;
  let backupScore = currentScore;
  let simulatedScore = 0;
  const originalUpdateScore = updateScore;
  updateScore = function(add) { simulatedScore += add; };

  tiles = JSON.parse(JSON.stringify(state));
  let moved = moveFunc();
  let newState = JSON.parse(JSON.stringify(tiles));

  updateScore = originalUpdateScore;
  currentScore = backupScore;
  tiles = backupTiles;

  return { moved, newState, simScore: simulatedScore };
}

// —— 在给定 state + (seed, step) 下，按规则添加确定性新砖（仅用于模拟）
function simulateDeterministicSpawn(state, seed, step) {
  const emptyCells = listEmptyCells(state);
  if (emptyCells.length === 0) return { state: state.slice(), nextStep: step }; // 满了
  const info = generateTileInfo(seed, step, side);
  const idx = info.emptySlotIndex % emptyCells.length;
  const pos = emptyCells[idx];
  const level = info.value === 2 ? 1 : 2;

  const newState = state.slice();
  // id 在模拟中无需真实唯一，仅保证字段存在
  newState.push({
    id: -1000000 - step, // 虚拟 id，避免与真实 id 冲突
    value: info.value,
    n: level,
    row: pos.r,
    col: pos.c,
    oldRow: pos.r,
    oldCol: pos.c,
    merged: false,
    spawned: true
  });
  return { state: newState, nextStep: step + 1 };
}

// —— 启发式评估：单调性 + 角落奖励 + 空格奖励 + 平滑度
function evaluateState(state) {
  if (state.length === 0) return 0;

  // 网格值矩阵
  const grid = Array.from({length: side}, () => Array(side).fill(0));
  let maxV = 0;
  for (const t of state) {
    grid[t.row][t.col] = t.value;
    if (t.value > maxV) maxV = t.value;
  }

  // 单调性（行/列方向上尽量单调）
  let mono = 0;
  for (let r = 0; r < side; r++) {
    for (let c = 1; c < side; c++) {
      mono += Math.sign(grid[r][c-1] - grid[r][c]) * Math.min(grid[r][c-1], grid[r][c]);
    }
  }
  for (let c = 0; c < side; c++) {
    for (let r = 1; r < side; r++) {
      mono += Math.sign(grid[r-1][c] - grid[r][c]) * Math.min(grid[r-1][c], grid[r][c]);
    }
  }

  // 空格数量
  const empties = listEmptyCells(state).length;

  // 平滑度（相邻差越小越好）
  let smooth = 0;
  const dirs = [[1,0],[0,1]];
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      if (!grid[r][c]) continue;
      for (const [dr,dc] of dirs) {
        let r2 = r+dr, c2 = c+dc;
        if (r2<side && c2<side && grid[r2][c2]) {
          smooth -= Math.abs(grid[r][c] - grid[r2][c2]);
        }
      }
    }
  }

  // 角落最大块奖励（偏左下角）
  const cornerBonus =
    (grid[side-1][0] === maxV ? 1 : 0) * maxV * 2 +
    (grid[side-1][1] === maxV ? 1 : 0) * (maxV * 0.2) +
    (grid[side-2]?.[0] === maxV ? 1 : 0) * (maxV * 0.2);

  // 加权合成
  const score =
    mono * 0.5 +
    smooth * 0.1 +
    cornerBonus +
    empties * (maxV * 0.02 + 50);

  return score;
}

// —— 方向映射（与现有 move* 对齐）：0:Up,1:Right,2:Down,3:Left
const MOVE_FUNCS = [moveUp, moveRight, moveDown, moveLeft];
const MOVE_ORDER = [2, 3, 1, 0]; // 优先尝试 Down/Left（常见策略）

// —— 置换表（Transposition Table）
const TT = new Map();
function hashState(state, step) {
  // 压成字符串：值-行-列|... + step
  const arr = state.slice().sort((a,b)=> (a.row*side+a.col) - (b.row*side+b.col))
    .map(t => `${t.value}:${t.row},${t.col}`).join("|");
  return arr + `#${step}`;
}

// —— Alpha-Beta 搜索结点，返回：{score, bestDir, pv:[方向序列]}
function searchNode(state, depth, alpha, beta, seed, step) {
  // 终局或深度到达
  // 是否无合法走法
  let anyMove = false;

  // 置换表命中
  const key = hashState(state, step) + `~${depth}`;
  const ttHit = TT.get(key);
  if (ttHit && ttHit.depth >= depth) {
    return { score: ttHit.score, bestDir: ttHit.bestDir, pv: ttHit.pv.slice() };
  }

  let bestScore = -Infinity;
  let bestDir = -1;
  let bestPV = [];

  // 移动排序：根据“立即得分 + 评估”做简单排序（可增益剪枝效果）
  const orderedMoves = MOVE_ORDER.slice();

  for (const dir of orderedMoves) {
    const sim = simulateMoveOnState(MOVE_FUNCS[dir], state);
    if (!sim.moved) continue;
    anyMove = true;

    // 移动后确定性生成新砖
    const afterMove = sim.newState;
    const { state: afterSpawn, nextStep } = simulateDeterministicSpawn(afterMove, seed, step);

    let nodeScore;
    let childPV = [];
    if (depth <= 1) {
      nodeScore = sim.simScore + evaluateState(afterSpawn);
    } else {
      // 递归
      const child = searchNode(afterSpawn, depth - 1, alpha - sim.simScore, beta - sim.simScore, seed, nextStep);
      nodeScore = sim.simScore + child.score;
      childPV = child.pv;
    }

    if (nodeScore > bestScore) {
      bestScore = nodeScore;
      bestDir = dir;
      bestPV = [dir, ...childPV];
    }

    // Alpha-Beta
    if (bestScore > alpha) alpha = bestScore;
    if (alpha >= beta) break; // 剪枝
  }

  if (!anyMove) {
    // 无法移动：终局评价（不要再加 spawn）
    bestScore = evaluateState(state) - 1e6; // 强惩罚
    bestDir = -1;
    bestPV = [];
  }

  TT.set(key, { depth, score: bestScore, bestDir, pv: bestPV });
  return { score: bestScore, bestDir, pv: bestPV };
}

// —— 迭代加深：给定最大深度，返回主变着（PV）
function iterativeDeepening(state, seed, step, maxDepth) {
  let best = { score: -Infinity, bestDir: -1, pv: [] };
  // 简单的迭代加深，不设置时间预算（需要的话可加入时间切片）
  for (let d = 2; d <= maxDepth; d++) {
    const res = searchNode(state, d, -Infinity, Infinity, seed, step);
    if (res.score > best.score) best = res;
    // 若 PV 为空，提前停止
    if (res.pv.length === 0) break;
  }
  return best;
}

/* ===== 自动游玩（以 0.1s/步播放“已确定的主线”） ===== */
let autoPlay = false;
let plannedMoves = [];  // 存储方向序列（0:Up,1:Right,2:Down,3:Left）
const MOVE_INTERVAL_MS = 100; // 0.1s/步

function computePlanAndQueue() {
  // 基于当前全局局面 + (seed, step) 搜索
  TT.clear();
  const state = cloneState(tiles);
  const plan = iterativeDeepening(state, globalSeed, globalStep, getSearchDepthByBoard());

  // 把得到的主变着（PV）入队；限制一次最多播放 16 步，避免长时间脱离 UI
  const batch = plan.pv.slice(0, 16);
  plannedMoves.push(...batch);

  // 可选：在控制台打印这段“确定的结果”
  if (batch.length) {
    console.log("确定主线（PV，批量）:", batch);
  }
}

// 根据棋盘大小自适应深度（越大棋盘，适当降低）
function getSearchDepthByBoard() {
  if (side <= 4) return 10;
  if (side === 5) return 9;
  if (side === 6) return 8;
  if (side === 7) return 7;
  if (side === 8) return 6;
  return 5;
}

function scheduleNextMove() {
  if (!autoPlay || !isGameStarted) return;
  if (plannedMoves.length === 0) {
    // 重新计算一段“确定主线”
    computePlanAndQueue();
    if (plannedMoves.length === 0) {
      autoPlay = false;
      return;
    }
  }

  const dir = plannedMoves.shift();
  if (dir == null) return;

  const moved = MOVE_FUNCS[dir]();
  if (moved) {
    renderAllTiles();
    setTimeout(() => {
      finalizePositions();
      spawnRandomTile();
      renderAllTiles();
      checkGameOver();
      if (autoPlay) {
        setTimeout(scheduleNextMove, MOVE_INTERVAL_MS);
      }
    }, 50);
  } else {
    // 非法或无效时，直接下一步/或重新规划
    setTimeout(scheduleNextMove, 0);
  }
}

// 绑定自动模式按钮（切换）
document.getElementById("btnAutoPlay").addEventListener("click", () => {
  autoPlay = !autoPlay;
  if (autoPlay) {
    plannedMoves.length = 0;
    computePlanAndQueue();
    scheduleNextMove();
  }
});

/* =========================
   结束：新 AI 模块
========================= */
