var canvasWidth = 400
var canvasHeight = 400
var bg_color = 245
var cursor_size = 20
var stroke_weight_ratio = 10
var frame_rate = 5
var jsonfile = 'actions.json'

var canvas
var played = false
var action = []
var prev_x = 0
var prev_y = 0
var data
var step
var step_length

function preload() {
	console.log('Loading ' + jsonfile)
	data = loadJSON(jsonfile)
}

function randint(n) {
	return Math.floor(Math.random() * Math.floor(n))
}

function sample_action() {
	var i = randint(data['actions'].length)
	var sampled_data = data['actions'][i].slice()

	// reset the timer
	step = 0

	return sampled_data
}

function setup() {
	canvas = createCanvas(canvasWidth, canvasHeight);
	canvas.parent('sketch-holder')
	frameRate(frame_rate)
}

function play() {
	// clear canvas
	clear()
	action = sample_action()
	step_length = action.length
	played = true
}


function draw() {
	// draw background
	background(bg_color)

	// draw border of the canvas
	stroke(0)
	strokeWeight(1)
	noFill()
	rect(0, 0, canvasWidth-1, canvasHeight-1)

	// draw contents
	if (played) {
		drawAction(action, step)
		step += 1
		if (step == step_length) {
			// reaches the terminal state. stop playing
			played = false
		}
	}
}

function drawAction(action, step) {
	// draw a cursor at the latest action
	if (step > 0) {
		drawCursor(action[step-1], 128)
	}

	drawCursor(action[step], 255)

	// draw sequence of lines until the indicated time step
	for (var t = 0; t < step; t++){
		drawLine(action[t])
	}
}

function drawCursor(action, alpha) {
	var [x, y, p, r, g, b, q] = action

	x = x * canvasWidth
	y = y * canvasHeight

	// draw cursor
	stroke(255,105,180, alpha)  // hot pink
	strokeWeight(4)
	ellipse(x, y, cursor_size, cursor_size)
}

function drawLine(action) {
	var [x, y, p, r, g, b, q] = action

	x = x * canvasWidth
	y = y * canvasHeight

	if (q) {
		// draw line
		strokeWeight(p * stroke_weight_ratio)
		stroke(r * 255, g * 255, b * 255)
		line(prev_x, prev_y, x, y)
	}
	
	// update the previous point
	prev_x = x
	prev_y = y
}
