// Variables for referencing the canvas and 2dcanvas context
var canvas, ctx;

// Variables to keep track of the mouse position and left-button status
var mouseX, mouseY, mouseDown = 0;

// Draws a dot at a specific position on the supplied canvas name
function drawDot(ctx, x, y) {
  // Draw a thicker line by setting multiple pixels (5x5 area)
  // Use Alpha = 160 for better contrast while matching EMNIST style
  for (let dx = -2; dx <= 2; dx++) {
    for (let dy = -2; dy <= 2; dy++) {
      const drawX = Math.round(x) + dx;
      const drawY = Math.round(y) + dy;

      // Check bounds
      if (drawX >= 0 && drawX < 28 && drawY >= 0 && drawY < 28) {
        let pxData = ctx.getImageData(drawX, drawY, 1, 1);
        pxData.data[0] = 0;    // R
        pxData.data[1] = 0;    // G
        pxData.data[2] = 0;    // B
        pxData.data[3] = 160;  // Alpha (better contrast, still EMNIST-like)
        ctx.putImageData(pxData, drawX, drawY);
      }
    }
  }
}

// Clear the canvas context using the canvas width and height
function clearCanvas(canvas, ctx) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Keep track of the mouse button being pressed and draw a dot at current location
function sketchpad_mouseDown() {
  mouseDown = 1;
  drawDot(ctx, mouseX / 5, mouseY / 5);
}

// Keep track of the mouse button being released
function sketchpad_mouseUp() {
  mouseDown = 0;
}

// Keep track of the mouse position and draw a dot if mouse button is currently pressed
function sketchpad_mouseMove(e) {
  // Update the mouse co-ordinates when moved
  getMousePos(e);

  // Draw a dot if the mouse button is currently being pressed
  if (mouseDown == 1) {
    drawDot(ctx, mouseX / 5, mouseY / 5);
  }
}

function reset() {
  clearCanvas(canvas, ctx);
}

// Get the current mouse position relative to the top-left of the canvas
function getMousePos(e) {
  if (!e)
    var e = event;

  if (e.offsetX) {
    mouseX = e.offsetX;
    mouseY = e.offsetY;
  } else if (e.layerX) {
    mouseX = e.layerX;
    mouseY = e.layerY;
  }
}

// Set-up the canvas and add our event handlers after the page has loaded
function init() {
  console.log('=== ИНИЦИАЛИЗАЦИЯ CANVAS ===');
  
  // Get the specific canvas element from the HTML document
  canvas = document.getElementById('sketchpad');
  console.log('1. Canvas element:', canvas);

  // If the browser supports the canvas tag, get the 2d drawing context for this canvas
  if (canvas.getContext) {
    ctx = canvas.getContext('2d', { willReadFrequently: true });
    console.log('2. Canvas context:', ctx);
    console.log('   Canvas size:', canvas.width, 'x', canvas.height);
    
    ctx.mozImageSmoothingEnabled = false;
    ctx.webkitImageSmoothingEnabled = false;
    ctx.msImageSmoothingEnabled = false;
    ctx.imageSmoothingEnabled = false;

    // Initialize with white background
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    console.log('3. Canvas cleared (white background with alpha=0)');
    
    // Check if background was set correctly
    const checkData = ctx.getImageData(0, 0, 1, 1);
    console.log('4. Проверка фона (первый пиксель):', checkData.data);
    console.log('   Alpha channel:', checkData.data[3], '(должно быть 0 для белого фона)');
  }

  // Check that we have a valid context to draw on/with before adding event handlers
  if (ctx) {
    canvas.addEventListener('mousedown', sketchpad_mouseDown, false);
    canvas.addEventListener('mousemove', sketchpad_mouseMove, false);
    window.addEventListener('mouseup', sketchpad_mouseUp, false);
    console.log('5. Event listeners added');
  }
  
  console.log('=== ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА ===');
}

function predictData() {
  console.log('=== НАЧАЛО ОБРАБОТКИ ИЗОБРАЖЕНИЯ ===');
  
  // Get image data from canvas
  const data = ctx.getImageData(0, 0, 28, 28);
  console.log('1. Получены данные canvas:', data);

  // Extract alpha channel (transparency) - this represents the drawing
  var img = [];
  for (let i = 0; i < 28 * 28; i++) {
    img.push(data.data[4 * i + 3]);
  }
  console.log('2. Извлечен альфа-канал:', img.slice(0, 10), '... (всего', img.length, 'значений)');
  console.log('   Мин/Макс значений:', Math.min(...img), '/', Math.max(...img));
  console.log('   Количество ненулевых пикселей:', img.filter(x => x > 0).length);

  // Convert to string format that backend expects: "[1,2,3,4,5]"
  const imageString = "[" + img.join(",") + "]";
  console.log('3. Преобразование в строку:', imageString.substring(0, 50) + '...');
  console.log('   Длина строки:', imageString.length);

  $.ajax({
    url: "/api/predict",
    type: "POST",
    headers: {
      "Accept": "application/json",
      "Content-Type": "application/json"
    },
    data: JSON.stringify(imageString),
    processData: false,
    success: (res, status) => {
      console.log("4. УСПЕШНЫЙ ОТВЕТ ОТ СЕРВЕРА:", res);
      console.log("   Предсказанный символ:", res.prediction);
      console.log("   ASCII код:", res.prediction.charCodeAt(0));
      
      const predCharCode = res.prediction.charCodeAt(0);
      const predChar = res.prediction;
      const isPrintable = predCharCode >= 32 && predCharCode <= 126;

      const resultText = isPrintable
        ? `Символ: '${predChar}' с кодом: ${predCharCode}`
        : `Непечатаемый символ с кодом: ${predCharCode}`;

      document.getElementById("prediction").innerHTML = `
        <div style="font-size: 20px; font-weight: bold; margin-top: 10px; padding: 10px; border: 2px solid #4CAF50; border-radius: 5px;">
          Предсказание: ${isPrintable ? predChar : `[${predCharCode}]`}<br>
          <span style="font-size: 14px; color: #666;">${resultText}</span>
        </div>
      `;
      console.log('=== КОНЕЦ ОБРАБОТКИ ИЗОБРАЖЕНИЯ ===');
    },
    error: (xhr, status, error) => {
      console.error("4. ОШИБКА ЗАПРОСА:", error);
      console.error("   Статус:", status);
      console.error("   Response text:", xhr.responseText);
      document.getElementById("prediction").innerHTML = `
        <div style="color: red; font-weight: bold;">
          Ошибка: ${error}. Проверьте консоль браузера.
        </div>
      `;
      console.log('=== КОНЕЦ ОБРАБОТКИ ИЗОБРАЖЕНИЯ (С ОШИБКОЙ) ===');
    }
  });
}
