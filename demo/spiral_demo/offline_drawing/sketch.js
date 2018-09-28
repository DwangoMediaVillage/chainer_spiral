var canvasWidth = 400
var canvasHeight = 400
var canvas

var frame_rate = 10
var played = false
var actions = []
var prev_x = 0
var prev_y = 0
var stroke_weight_ratio = 10
var data
var jsonfile = 'foo.json'

function preload() {
	console.log('Loading ' + jsonfile)
	data = loadJSON(jsonfile)
}

function setup() {
	canvas = createCanvas(canvasWidth, canvasHeight);
	canvas.parent('sketch-holder')
	frameRate(frame_rate)

	actions = data['actions'].slice()
}

function start() {
	played = true
}

function stop() {
	played = false
}

function reset() {
	// clear canvas and reset status
	clear()
	played = false
	actions = data['actions'].slice()
}

function draw() {
	// draw border of the canvas
	stroke(0)
	strokeWeight(1)
	noFill()
	rect(0, 0, canvasWidth-1, canvasHeight-1)

	// draw contents
	if (played) {
		var action = actions.shift()
		if (action) {
			drawLine(action)
		} else {
			played = false
		}
	}
}

function drawLine(action) {
	var [x, y, p, r, g, b, q] = action

	// scale x and y
	x = x * canvasWidth
	y = y * canvasHeight

	if (q) {
		// draw
		strokeWeight(p * stroke_weight_ratio)
		stroke(r * 255, g * 255, b * 255)
		line(prev_x, prev_y, x, y)
	}
	
	// update the previous point
	prev_x = x
	prev_y = y
}
