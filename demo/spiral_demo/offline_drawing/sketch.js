const jsonfile = 'actions.json'
const bgColor = 245
const borderColor = 255
const cursorSize = 4
const strokeWeightRatio = 5.0
const borderWeight = 3
var graphicsSize = 100
const frameRateVal = 10
const nRows = 3
const nCols = 5
const buttonOffset = 30

var canvas
var data
var graphics = []
var t = 0
var actions = []
var steps = []
var played = false
var button

function preload() {
	console.log('Loading ' + jsonfile)
	data = loadJSON(jsonfile)
}

function randint(n) {
	return Math.floor(Math.random() * Math.floor(n))
}

function setup() {
	graphicsSize += borderWeight * 2
	canvas = createCanvas(graphicsSize * nCols, graphicsSize * nRows + buttonOffset)
	canvas.parent('sketch-holder')
	frameRate(frameRateVal)

	// put button to start playing
	button = createButton('Play')
	button.parent('sketch-holder')
	button.position(0, 0)
	button.mousePressed(play)

	// set and init graphics
	for (let i=0; i<nRows; i++) {
		for (let j=0; j<nCols; j++){
			let g = createGraphics(graphicsSize, graphicsSize)
			initGraphic(g)
			image(g, j * graphicsSize, i * graphicsSize + buttonOffset)
			graphics.push(g)
		}
	}
}


function draw() {
	background(borderColor);
	let n = 0
	let stopped = 0
	for (let i=0; i<nRows; i++) {
		for (let j=0; j<nCols; j++){
			let g = graphics[nCols * i + j]

			// update graphic
			if (played) {
				// draw background and border
				initGraphic(g)
				// update graphic
				stopped += updateGraphic(g, actions[n], steps[n], t)
			}
			
			// render graphics to the canvas
			image(g, j * graphicsSize, i * graphicsSize + buttonOffset)
			n += 1
		}
	}

	// update time counter
	t += 1

	// if all the graphics finish to draw, stop playing
	if (stopped == n) {
		console.log('playing stopped')
		played = false
	}
}

function play() {
	// sample action sequences from json buffer, and starts playing
	clear()
	actions = []
	steps = []

	t = 0
	for (let i=0; i < nRows * nCols; i++) {
		let action = sampleAction()
		actions.push(action)
		steps.push(action.length)
	}

	played = true
}

function sampleAction() {
	let i = randint(data['actions'].length)
	return data['actions'][i].slice()
}

function initGraphic(graphic) {
	// draw background and border of a grahics
	graphic.background(bgColor)

	// draw border of the canvas
	graphic.stroke(borderColor)
	graphic.strokeWeight(borderWeight)
	graphic.noFill()
	graphic.rect(0, 0, graphicsSize-borderWeight, graphicsSize-borderWeight)
}

function updateGraphic(graphic, action, steps, t) {
	// may be update a graphic, and returns 1 if step t reaches the terminal state
	
	// draw cursor
	if (t > 0) {
		drawCursor(graphic, action[t-1], 128)
	}
	drawCursor(graphic, action[t], 255)

	let tTerminate = 0
	if (t >= steps - 1) {
		tTerminate = steps - 1
	} else {
		tTerminate = t
	}

	// draw lines
	drawLines(graphic, action, tTerminate)

	return t >= steps - 1
}


function drawCursor(graphic, action, alpha) {
	let x = action[0],
		y = action[1]
	x = scalePoint(x)
	y = scalePoint(y)
	graphic.stroke(255, 105, 180, alpha)
	graphic.strokeWeight(4)
	graphic.ellipse(x, y, cursorSize, cursorSize)
}

function drawLines(graphic, action, tTerminate) {
	let prevX = 0
	let prevY = 0
	for (let t = 0; t < tTerminate; t++) {
		let [x, y, p, r, g, b, q] = action[t]
		x = scalePoint(x)
		y = scalePoint(y)
		if (q) {
			graphic.strokeWeight(p * strokeWeightRatio)
			graphic.stroke(r * 255, g * 255, b * 255)
			graphic.line(prevX, prevY, x, y)
		}
		prevX = x
		prevY = y
	}
}

function scalePoint(x) {
	return x * (graphicsSize - borderWeight * 2) + borderWeight
}