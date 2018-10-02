var canvasWidth = 400
var canvasHeight = 400
var canvas

var frame_rate = 10
var played = false
var action = []
var prev_x = 0
var prev_y = 0
var stroke_weight_ratio = 10
var data
var jsonfile = 'actions.json'

function preload() {
	console.log('Loading ' + jsonfile)
	data = loadJSON(jsonfile)
}

function randint(n) {
	return Math.floor(Math.random() * Math.floor(n))
}

function sample_action() {
	var i = randint(data['actions'].length)
	return  data['actions'][i].slice()
}

function setup() {
	canvas = createCanvas(canvasWidth, canvasHeight);
	canvas.parent('sketch-holder')
	frameRate(frame_rate)
	action = sample_action()
}

function play() {
	// clear canvas
	clear()
	played = true
}


function draw() {
	// draw border of the canvas
	stroke(0)
	strokeWeight(1)
	noFill()
	rect(0, 0, canvasWidth-1, canvasHeight-1)

	// draw contents
	if (played) {
		var act = action.shift()
		if (act) {
			console.log(act)
			drawLine(act)
		} else {
			played = false
			action = sample_action()
		}
	}
}

function drawLine(action) {
	var [x, y, p, r, g, b, q] = action

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
