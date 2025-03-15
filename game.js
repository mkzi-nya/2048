// Version: 1
// 本代码实现：自动游玩时，如果棋盘边长 < 7 则使用 worker.js 算法，否则使用预搜索算法

// ============ 全局变量 ============
let side = 4;
let base = 2;  // 基数，可调范围 2-10
let tiles = [];        // 当前所有方块，每个对象格式：{id, value, n, row, col, oldRow, oldCol, merged, spawned, mergedFrom}
let tileID = 0;        // 用于分配唯一 ID
let currentScore = 0;
let bestScore = 0;
let isGameStarted = false; // 游戏开始后才能操作

const SWIPE_THRESHOLD = 10;

// AI 相关变量：worker.js 版本需要
let aiWorker = new Worker("worker.js");
let aiStrength = 8;  // AI 强度（用于 worker 算法的概率阈值映射）

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




// ===== getMinProb：返回 worker 算法所用的最小概率阈值 =====
function getMinProb() {
  const probMap = {
    1: 0.02,
    2: 0.01,
    3: 0.005,
    4: 0.003,
    5: 0.002,
    6: 0.001,
    7: 0.0005,
    8: 0.0003,
    9: 0.0002,
    10: 0.0001,
  };
  return probMap[aiStrength] || 0.001;
}

// ===== executeMove：根据方向执行移动 =====
// 方向 0: Up, 1: Right, 2: Down, 3: Left
function executeMove(direction) {
  let moveFunctions = [moveUp, moveRight, moveDown, moveLeft];
  let moveFunc = moveFunctions[direction];
  if (moveFunc) {
    moveFunc();
  }
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

// ============ 生成随机方块 ============
function spawnRandomTile() {
  let emptyCells = [];
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      if (!tiles.some(t => t.row === r && t.col === c)) {
        emptyCells.push({ r, c });
      }
    }
  }
  if (emptyCells.length === 0) return;
  let pos = emptyCells[Math.floor(Math.random() * emptyCells.length)];

  let chosenLevel = 1;  // 默认生成 n1
  if (side <= 5) {
    // 棋盘较小时，80% n1，20% n2
    chosenLevel = (Math.random() < 0.8) ? 1 : 2;
  } else {
    // 棋盘大于 5 时，允许生成 n1 ~ (当前最大 tile 的等级 - 2)，最小为 n1
    // 先确定当前最大 tile 的等级
    let currentMax = base;
    if (tiles.length > 0) {
      currentMax = Math.max(...tiles.map(t => t.value));
    }
    let maxLevel = 1;
    if (currentMax > base) {
      maxLevel = Math.floor(Math.log(currentMax / base) / Math.log(2)) + 1;
    }
    let allowedMax = maxLevel - 2;
    if (allowedMax < 1) allowedMax = 1;

    // 构建等级概率分布：n1 固定 80%，n2 固定 10%，n3 开始每一级为前一级的一半，最低不低于 0.0002
    let probs = {};
    probs[1] = 0.8;
    if (allowedMax >= 2) {
      probs[2] = 0.1;
    }
    for (let i = 3; i <= allowedMax; i++) {
      let p = probs[i - 1] / 2;
      if (p < 0.0002) {
        p = 0.0002;
      }
      probs[i] = p;
    }
    // 归一化概率
    let total = 0;
    for (let i = 1; i <= allowedMax; i++) {
      total += probs[i];
    }
    let r = Math.random() * total;
    let cumulative = 0;
    for (let i = 1; i <= allowedMax; i++) {
      cumulative += probs[i];
      if (r < cumulative) {
        chosenLevel = i;
        break;
      }
    }
  }

  // 计算 tile 数值： value = base * 2^(level-1)
  let chosenValue = base * Math.pow(2, chosenLevel - 1);
  tiles.push({
    id: ++tileID,
    value: chosenValue,
    n: chosenLevel,
    row: pos.r,
    col: pos.c,
    oldRow: pos.r,
    oldCol: pos.c,
    merged: false,
    spawned: true
  });
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
      darkMode: document.body.classList.contains("dark-mode")
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

// ======== 以下为 AI 自动游玩及模拟预搜索算法 ============
let autoPlay = false;
let simulateMode = false;

/**
 * 在不改变全局状态下模拟一次移动
 * 修改点：在模拟过程中临时覆盖 updateScore，
 * 捕获移动过程中加分（merge 时调用 updateScore 产生的分数），
 * 模拟结束后恢复全局 tiles 与 currentScore，不影响界面显示的分数。
 */
function simulateMoveOnState(moveFunc, state) {
  // 备份全局状态
  let backupTiles = tiles;
  let backupScore = currentScore;
  let simulatedScore = 0;
  // 临时重写 updateScore，不更新界面，而是累计分数
  const originalUpdateScore = updateScore;
  updateScore = function(add) {
    simulatedScore += add;
  };

  tiles = JSON.parse(JSON.stringify(state));
  let moved = moveFunc();
  let newState = JSON.parse(JSON.stringify(tiles));

  // 恢复 updateScore 与全局 currentScore、tiles
  updateScore = originalUpdateScore;
  currentScore = backupScore;
  tiles = backupTiles;

  return { moved, newState, simScore: simulatedScore };
}

// 评价状态：越靠近左下角得分越高；左下角放置最大数额外奖励
function evaluateState(state) {
  let score = 0;
  for (let row = 0; row < side; row++) {
    for (let col = 0; col < side; col++) {
      let tile = state.find(t => t.row === row && t.col === col);
      if (tile) {
        let distanceFromBottomLeft = (side - row) * 10 + (side - col);
        score += tile.value * distanceFromBottomLeft;
      }
    }
  }
  let largestTile = Math.max(...state.map(t => t.value));
  let bottomLeftTile = state.find(t => t.row === side - 1 && t.col === 0);
  if (bottomLeftTile && bottomLeftTile.value === largestTile) {
    score += 5000;
  }
  return score;
}

// 递归搜索：预搜索 depth 层后返回最佳评分
function search(depth, state) {
  if (depth === 0) {
    return evaluateState(state);
  }
  let bestScore = -Infinity;
  let moves = [moveDown, moveLeft, moveRight, moveUp];
  for (let moveFunc of moves) {
    let sim = simulateMoveOnState(moveFunc, state);
    if (!sim.moved) continue;
    let score = sim.simScore + search(depth - 1, sim.newState);
    if (score > bestScore) {
      bestScore = score;
    }
  }
  return bestScore;
}

// 选择最佳移动（模拟算法）
function findBestMove() {
  let bestMove = null;
  let bestScore = -Infinity;
  let moves = [
    { move: moveDown },
    { move: moveLeft },
    { move: moveRight },
    { move: moveUp },
  ];
  const depth = 5;
  for (let candidate of moves) {
    let sim = simulateMoveOnState(candidate.move, tiles);
    if (!sim.moved) continue;
    let score = sim.simScore + search(depth - 1, sim.newState);
    if (score > bestScore) {
      bestScore = score;
      bestMove = candidate.move;
    }
  }
  return bestMove;
}

// 自动移动函数：根据棋盘大小选择算法
function autoMove() {
  if (!autoPlay || !isGameStarted) return;
  
  if (side < 7) {
    // 使用 worker.js 算法
    autoMoveWorker();
  } else {
    // 使用预搜索模拟算法
    let bestMove = findBestMove();
    if (bestMove) {
      bestMove(); // 执行最佳移动
      renderAllTiles();
      finalizePositions();
      spawnRandomTile();
      renderAllTiles();
      checkGameOver();
      setTimeout(autoMove, 0);
    } else {
      autoPlay = false;
    }
  }
}

// worker.js 算法分支：异步调用 worker
function autoMoveWorker() {
  let gridArray = new Uint32Array(side * side);
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      let tile = tiles.find(t => t.row === r && t.col === c);
      gridArray[r * side + c] = tile ? tile.value : 0;
    }
  }
  aiWorker.postMessage({ grid: gridArray, minProb: getMinProb() });
  aiWorker.onmessage = function(event) {
    let bestDirection = event.data;
    if (bestDirection !== null && bestDirection !== undefined) {
      executeMove(bestDirection);
      renderAllTiles();
      finalizePositions();
      spawnRandomTile();
      renderAllTiles();
      checkGameOver();
      setTimeout(autoMove, 0);
    } else {
      autoPlay = false;
    }
  };
}

// 绑定自动模式按钮
document.getElementById("btnAutoPlay").addEventListener("click", () => {
  autoPlay = !autoPlay;
  if (autoPlay) {
    autoMove();
  }
});
