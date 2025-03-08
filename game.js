// ============ 全局变量 ============

// 默认 4x4
let side = 4;
// 所有可能的最大随机值
let maxTileCandidates = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384];
// 默认最大值指向 4
let maxTileIndex = 1;

let tiles = [];        // 当前所有方块，每个对象格式：{id, value, row, col, oldRow, oldCol, merged, spawned, mergedFrom}
let tileID = 0;        // 用来给每个方块分配唯一ID
let currentScore = 0;
let bestScore = 0;
let isGameStarted = false; // 只有开始游戏后才可操作

// ================== 調整滑動靈敏度 ==================
const SWIPE_THRESHOLD = 10;

// ============ 页面加载 ============
window.addEventListener("DOMContentLoaded", () => {
  // 从 localStorage 读取最高分
  if(localStorage.getItem("bestScore")) {
    bestScore = parseInt(localStorage.getItem("bestScore"));
    document.getElementById("bestScore").innerText = bestScore;
  }

  // 初始化面板显示
  document.getElementById("spanSide").innerText = side;
  document.getElementById("spanMaxTile").innerText = maxTileCandidates[maxTileIndex];

  // 生成一个初始背景格（仅显示背景，不生成方块）
  createGrid(side);

  // 绑定底部调节按钮
  document.getElementById("btnSideMinus").addEventListener("click", () => changeSide(-1));
  document.getElementById("btnSidePlus").addEventListener("click", () => changeSide(+1));
  document.getElementById("btnMaxMinus").addEventListener("click", () => changeMaxTile(-1));
  document.getElementById("btnMaxPlus").addEventListener("click", () => changeMaxTile(+1));

  document.getElementById("btnApply").addEventListener("click", applySettings);
  document.getElementById("btnRestart").addEventListener("click", restartGame);
  document.getElementById("btnDarkMode").addEventListener("click", toggleDarkMode);

  // 只在網格容器上禁止觸摸滾動
  const gridContainer = document.getElementById("gridContainer");
  gridContainer.addEventListener("touchmove", (e)=> e.preventDefault(), {passive:false});
});

// ============ 创建背景格子 ============
function createGrid(s) {
  const gridContainer = document.getElementById("gridContainer");
  gridContainer.innerHTML = "";

  // 设置网格行列
  gridContainer.style.gridTemplateRows = `repeat(${s}, 1fr)`;
  gridContainer.style.gridTemplateColumns = `repeat(${s}, 1fr)`;

  // 生成背景格
  for(let i = 0; i < s; i++){
    for(let j = 0; j < s; j++){
      const cell = document.createElement("div");
      cell.classList.add("grid-cell");
      gridContainer.appendChild(cell);
    }
  }
}

// ============ 改变边长 ============
function changeSide(delta) {
  if(isGameStarted) return; // 游戏开始后不允许调整
  let newVal = side + delta;
  if(newVal < 2) newVal = 2;
  if(newVal > 10) newVal = 10;
  side = newVal;
  document.getElementById("spanSide").innerText = side;
  createGrid(side);
}

// ============ 改变最大随机值 ============
function changeMaxTile(delta) {
  if(isGameStarted) return;
  maxTileIndex += delta;
  if(maxTileIndex < 0) maxTileIndex = 0;
  if(maxTileIndex >= maxTileCandidates.length) {
    maxTileIndex = maxTileCandidates.length - 1;
  }
  document.getElementById("spanMaxTile").innerText = maxTileCandidates[maxTileIndex];
}

// ============ 点击“开始游戏” ============
function applySettings() {
  if(isGameStarted) return;
  isGameStarted = true;
  // 禁用调节按钮，开始游戏后不能修改
  document.getElementById("btnSideMinus").disabled = true;
  document.getElementById("btnSidePlus").disabled = true;
  document.getElementById("btnMaxMinus").disabled = true;
  document.getElementById("btnMaxPlus").disabled = true;
  document.getElementById("btnApply").disabled = true;

  setupGame();
}

// ============ 初始化/重置游戏 ============
function setupGame() {
  // 重置分数
  currentScore = 0;
  document.getElementById("currentScore").innerText = "0";

  // 清空方块数据
  tiles = [];
  tileID = 0;

  // 重建背景格
  createGrid(side);

  // 随机生成2个初始方块（可不做特效）
  spawnRandomTile();
  spawnRandomTile();

  // 绑定输入事件（键盘和滑动）
  initInputEvents();

  // 首次渲染时不需要动画
  renderAllTiles(true);
}

// ============ 重新开始 ============
function restartGame() {
  // 允许重新调整：恢复调节按钮状态
  isGameStarted = false;
  document.getElementById("btnSideMinus").disabled = false;
  document.getElementById("btnSidePlus").disabled = false;
  document.getElementById("btnMaxMinus").disabled = false;
  document.getElementById("btnMaxPlus").disabled = false;
  document.getElementById("btnApply").disabled = false;

  // 清空分数与方块数据
  tiles = [];
  tileID = 0;
  currentScore = 0;
  document.getElementById("currentScore").innerText = "0";

  // 移除 DOM 中的所有方块
  const oldTiles = document.querySelectorAll(".tile");
  oldTiles.forEach(t => t.remove());

  // 重建背景格
  createGrid(side);
}

// ============ 暗色模式切换 ============
function toggleDarkMode() {
  document.body.classList.toggle("dark-mode");
  localStorage.setItem("darkMode", document.body.classList.contains("dark-mode"));
}

// 页面加载时检查是否启用暗色模式
window.addEventListener("DOMContentLoaded", () => {
  if (localStorage.getItem("darkMode") === "true") {
    document.body.classList.add("dark-mode");
  }
});


// ============ 分数更新 ============
function updateScore(add) {
  currentScore += add;
  document.getElementById("currentScore").innerText = currentScore;
  if(currentScore > bestScore) {
    bestScore = currentScore;
    document.getElementById("bestScore").innerText = bestScore;
    localStorage.setItem("bestScore", bestScore);
  }
}

// ============ 输入事件（键盘+滑动） ============

let inputInited = false;
function initInputEvents() {
  if(inputInited) return;
  inputInited = true;

  // 鍵盤事件
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
      // 先執行移動動畫
      renderAllTiles();

      // 0.2 秒後：更新 oldRow/oldCol、生成新方塊、再次渲染、判斷遊戲結束
      setTimeout(() => {
        finalizePositions();
        spawnRandomTile();
        renderAllTiles();
        checkGameOver();
      }, 200);
    }
  });

  // 觸摸事件
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

// 將所有方塊的 oldRow/oldCol 更新為當前位置，並清除 merged 狀態
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
 * 將一行 / 一列的方塊按照方向進行合併
 * lineTiles 為同一行或同一列、且已按壓縮方向排序好的陣列
 */
function compressLine(lineTiles, direction) {
  if(lineTiles.length === 0) return false;

  let changed = false;
  let newLine = [];
  let skip = false;

  for (let i = 0; i < lineTiles.length; i++) {
    if(skip) {
      skip = false;
      continue;
    }
    // 檢查是否可合併
    if(i < lineTiles.length - 1 && lineTiles[i].value === lineTiles[i+1].value) {
      let targetTile = lineTiles[i];
      let mergingTile = lineTiles[i+1];
      targetTile.value *= 2;
      updateScore(targetTile.value);

      targetTile.merged = true;
      // 記錄合併來源，用於渲染動畫時從 mergingTile 的舊位置飛過來
      targetTile.mergedFrom = { row: mergingTile.row, col: mergingTile.col };
      // 強制將合併後 tile 的 oldRow/oldCol 設成被合併的那個位置
      targetTile.oldRow = mergingTile.row;
      targetTile.oldCol = mergingTile.col;

      newLine.push(targetTile);
      // 移除被合併的 tile
      tiles = tiles.filter(x => x.id !== mergingTile.id);
      skip = true;
      changed = true;
    } else {
      newLine.push(lineTiles[i]);
    }
  }

  // 根據方向重排最終的 row/col
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
      if(!tiles.some(t => t.row === r && t.col === c)) {
        emptyCells.push({ r, c });
      }
    }
  }
  if(emptyCells.length === 0) return;

  let maxRandomVal = maxTileCandidates[maxTileIndex];
  let allVals = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384];
  let allProb = [0.88,0.1,0.01,0.005,0.002,0.001,0.0005,0.00025,0.00025,0.0002,0.00005,0.00003,0.00002,0.00002];
  let vals = [], probs = [];
  for(let i = 0; i < allVals.length; i++){
    if(allVals[i] <= maxRandomVal){
      vals.push(allVals[i]);
      probs.push(allProb[i]);
    }
  }
  let sumP = probs.reduce((a, b) => a + b, 0);
  let norm = probs.map(x => x / sumP);
  let rand = Math.random(), cum = 0, chosen = vals[0];
  for(let i = 0; i < norm.length; i++){
    cum += norm[i];
    if(rand <= cum) {
      chosen = vals[i];
      break;
    }
  }

  let pos = emptyCells[Math.floor(Math.random() * emptyCells.length)];
  tiles.push({
    id: ++tileID,
    value: chosen,
    row: pos.r,
    col: pos.c,
    oldRow: pos.r,
    oldCol: pos.c,
    merged: false,
    spawned: true  // 用於生成動畫
  });
}

// ============ 渲染方块（含平滑移动与弹出动画） ============
function renderAllTiles(skipAnimation = false) {
  const container = document.getElementById("gridContainer");
  const bgCells = container.querySelectorAll(".grid-cell");

  // 移除旧的 DOM 方块
  const oldDOM = container.querySelectorAll(".tile");
  oldDOM.forEach(d => d.remove());

  // 根據現在的 tiles 狀態，重新生成 DOM
  tiles.forEach(t => {
    const div = document.createElement("div");
    div.classList.add("tile", `tile-${t.value}`);

    // 調整動畫時間：生成 0.1s，合併 0.2s，其餘也 0.2s
    if(t.spawned) {
      div.classList.add("tile-pop");
      div.style.transitionDuration = "0.1s";
    } else if(t.merged) {
      div.classList.add("tile-merged");
      div.style.transitionDuration = "0.2s";
    } else {
      div.style.transitionDuration = "0.2s";
    }

    div.innerText = t.value;

    // 初始位置（視覺上）先放在 oldRow/oldCol
    let oldRect = getCellRect(bgCells, t.oldRow, t.oldCol);
    div.style.width = oldRect.width + "px";
    div.style.height = oldRect.height + "px";
    div.style.left = oldRect.left + "px";
    div.style.top = oldRect.top + "px";
    div.style.fontSize = Math.floor(oldRect.width * 0.4) + "px";

    container.appendChild(div);
  });

  // 首次渲染或 skipAnimation 時，直接跳到新位置，不播放動畫
  if(skipAnimation) {
    tiles.forEach(t => {
      t.oldRow = t.row;
      t.oldCol = t.col;
    });
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

  // 否則啟動 CSS transition，讓方塊從舊位置平移到新位置
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

  // 清除新生成標記，確保生成動畫只播一次
  tiles.forEach(t => { t.spawned = false; });
}

// 輔助：計算某行 row、某列 col 的背景方塊相對 container 的位置與大小
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
  // 如果还有空格子，则未结束
  if(tiles.length < side * side) return;
  // 检查是否有可合并的相邻同值方塊
  for(let i = 0; i < tiles.length; i++){
    for(let j = i + 1; j < tiles.length; j++){
      let a = tiles[i], b = tiles[j];
      if(a.value === b.value) {
        if(a.row === b.row && Math.abs(a.col - b.col) === 1) return;
        if(a.col === b.col && Math.abs(a.row - b.row) === 1) return;
      }
    }
  }
  alert("游戏结束！最终分数：" + currentScore);
}
