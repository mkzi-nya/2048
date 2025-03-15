// worker.js: Pure JS Expectimax AI for 2048

function cloneGrid(grid) {
  return grid.slice();
}

function getDim(grid) {
  return Math.sqrt(grid.length) | 0;
}

function compressRow(row, dim) {
  let arr = row.filter(x => x !== 0);
  for (let i = 0; i < arr.length - 1; i++) {
    if (arr[i] === arr[i + 1]) {
      arr[i] = arr[i] * 2;
      arr[i + 1] = 0;
      i++;
    }
  }
  arr = arr.filter(x => x !== 0);
  while (arr.length < dim) {
    arr.push(0);
  }
  return arr;
}

function moveLeft(grid) {
  let dim = getDim(grid);
  let newGrid = new Uint32Array(grid.length);
  let moved = false;
  for (let r = 0; r < dim; r++) {
    let row = [];
    for (let c = 0; c < dim; c++) {
      row.push(grid[r * dim + c]);
    }
    let newRow = compressRow(row, dim);
    for (let c = 0; c < dim; c++) {
      newGrid[r * dim + c] = newRow[c];
    }
    for (let c = 0; c < dim; c++) {
      if (row[c] !== newRow[c]) {
        moved = true;
      }
    }
  }
  return { newGrid: newGrid, moved: moved };
}

function moveRight(grid) {
  let dim = getDim(grid);
  let reversedGrid = new Uint32Array(grid.length);
  // 将每一行反转
  for (let r = 0; r < dim; r++) {
    for (let c = 0; c < dim; c++) {
      reversedGrid[r * dim + c] = grid[r * dim + (dim - 1 - c)];
    }
  }
  let result = moveLeft(reversedGrid);
  let newGrid = new Uint32Array(grid.length);
  // 再将结果反转回来
  for (let r = 0; r < dim; r++) {
    for (let c = 0; c < dim; c++) {
      newGrid[r * dim + c] = result.newGrid[r * dim + (dim - 1 - c)];
    }
  }
  return { newGrid: newGrid, moved: result.moved };
}

function transpose(grid) {
  let dim = getDim(grid);
  let newGrid = new Uint32Array(grid.length);
  for (let r = 0; r < dim; r++) {
    for (let c = 0; c < dim; c++) {
      newGrid[c * dim + r] = grid[r * dim + c];
    }
  }
  return newGrid;
}

function moveUp(grid) {
  let transposed = transpose(grid);
  let result = moveLeft(transposed);
  let newGrid = transpose(result.newGrid);
  return { newGrid: newGrid, moved: result.moved };
}

function moveDown(grid) {
  let transposed = transpose(grid);
  let result = moveRight(transposed);
  let newGrid = transpose(result.newGrid);
  return { newGrid: newGrid, moved: result.moved };
}

function move(grid, direction) {
  // 方向：0：上, 1：右, 2：下, 3：左
  switch (direction) {
    case 0: return moveUp(grid);
    case 1: return moveRight(grid);
    case 2: return moveDown(grid);
    case 3: return moveLeft(grid);
    default: return { newGrid: grid, moved: false };
  }
}

function validMoves(grid) {
  let moves = [];
  for (let d = 0; d < 4; d++) {
    let res = move(grid, d);
    if (res.moved) {
      moves.push(d);
    }
  }
  return moves;
}

function evaluate(grid) {
  let dim = getDim(grid);
  let empty = 0;
  let maxTile = 0;
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] === 0) empty++;
    if (grid[i] > maxTile) maxTile = grid[i];
  }
  return empty * 100 + maxTile;
}

function isTerminal(grid) {
  return validMoves(grid).length === 0;
}

function expectimax(grid, depth, isPlayerTurn, prob, minProb) {
  if (prob < minProb) return evaluate(grid);
  if (depth === 0 || isTerminal(grid)) return evaluate(grid);
  let dim = getDim(grid);
  if (isPlayerTurn) {
    let best = -Infinity;
    let moves = validMoves(grid);
    if (moves.length === 0) return evaluate(grid);
    for (let d of moves) {
      let res = move(grid, d);
      let score = expectimax(res.newGrid, depth - 1, false, prob, minProb);
      if (score > best) best = score;
    }
    return best;
  } else {
    // chance node：在空位上生成 2（0.9）或 4（0.1）
    let emptyCells = [];
    for (let i = 0; i < grid.length; i++) {
      if (grid[i] === 0) emptyCells.push(i);
    }
    if (emptyCells.length === 0) return evaluate(grid);
    let total = 0;
    for (let idx of emptyCells) {
      for (let tileValue of [2, 4]) {
        let tileProb = (tileValue === 2 ? 0.9 : 0.1);
        let newGrid = cloneGrid(grid);
        newGrid[idx] = tileValue;
        let score = expectimax(newGrid, depth - 1, true, prob * (tileProb / emptyCells.length), minProb);
        total += (tileProb / emptyCells.length) * score;
      }
    }
    return total;
  }
}

function chooseBestMove(grid, minProb, depth) {
  let moves = validMoves(grid);
  if (moves.length === 0) return null;
  let bestScore = -Infinity;
  let bestMove = null;
  for (let d of moves) {
    let res = move(grid, d);
    let score = expectimax(res.newGrid, depth - 1, false, 1, minProb);
    if (score > bestScore) {
      bestScore = score;
      bestMove = d;
    }
  }
  return bestMove;
}

onmessage = function (event) {
  let data = event.data;
  let grid = data.grid;
  let minProb = data.minProb;
  // 设定搜索深度，可根据需要调整（此处设置为 3）
  let depth = 4;
  let bestMove = chooseBestMove(grid, minProb, depth);
  postMessage(bestMove);
};
