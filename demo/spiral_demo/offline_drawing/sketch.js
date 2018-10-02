var jsonfile = 'actions.json'
var bg_color = 245
var cursor_size = 4
var stroke_weight_ratio = 5.0
var border_weight = 3
var graphics_size = 100
var frame_rate = 10
var n_rows = 5
var n_cols = 5

var canvas
var data
var graphics = []
var t = 0
var actions = []
var steps = []
var played = false

function preload() {
	console.log('Loading ' + jsonfile)
	data = loadJSON(jsonfile)
}

function randint(n) {
	return Math.floor(Math.random() * Math.floor(n))
}

function setup() {

	graphics_size = graphics_size + border_weight * 2

	canvas = createCanvas(graphics_size * n_rows, graphics_size * n_cols)
	canvas.parent('sketch-holder')
	frameRate(frame_rate)

	// set graphics
	for (var i=0; i<n_rows; i++) {
		for (var j=0; j<n_cols; j++){
			var g = createGraphics(graphics_size, graphics_size)
			init_graphic(g)
			image(g, i * graphics_size, j * graphics_size)
			graphics.push(g)
		}
	}
}


function draw() {
	background(255);
	var n = 0
	var stopped = 0
	for (var i=0; i<n_rows; i++) {
		for (var j=0; j<n_cols; j++){
			var g = graphics[n_cols * i + j]

			// update graphic
			if (played) {
				// draw background and border
				init_graphic(g)

				// update graphic
				stopped += update_graphic(g, actions[n], steps[n], t)
			}
			
			// render graphics to the canvas
			image(g, i * graphics_size, j * graphics_size)
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
	for (var i=0; i < n_rows * n_cols; i++) {
		var action = sample_action()
		actions.push(action)
		steps.push(action.length)
	}

	played = true
}

function sample_action() {
	var i = randint(data['actions'].length)
	return data['actions'][i].slice()
}

function init_graphic(graphic) {
	// draw background and border of a grahics
	graphic.background(bg_color)

	// draw border of the canvas
	graphic.stroke(255)
	graphic.strokeWeight(border_weight)
	graphic.noFill()
	graphic.rect(0, 0, graphics_size-border_weight, graphics_size-border_weight)
}

function update_graphic(graphic, action, steps, t) {
	// may be update a graphic, and returns 1 if step t reaches the terminal state
	
	// draw cursor
	if (t > 0) {
		drawCursor(graphic, action[t-1], 128)
	}
	drawCursor(graphic, action[t], 255)

	var t_terminate = 0
	if (t >= steps - 1) {
		t_terminate = steps - 1
	} else {
		t_terminate = t
	}

	// draw lines
	drawLines(graphic, action, t_terminate)

	return t >= steps - 1
}


function drawCursor(graphic, action, alpha) {
	var [x, y, p, r, g, b, q] = action
	x = scale_point(x)
	y = scale_point(y)
	graphic.stroke(255, 105, 180, alpha)
	graphic.strokeWeight(4)
	graphic.ellipse(x, y, cursor_size, cursor_size)
}

function drawLines(graphic, action, t_terminate) {
	var prev_x = 0
	var prev_y = 0
	for (var t = 0; t < t_terminate; t++) {
		var [x, y, p, r, g, b, q] = action[t]
		x = scale_point(x)
		y = scale_point(y)
		if (q) {
			graphic.strokeWeight(p * stroke_weight_ratio)
			graphic.stroke(r * 255, g * 255, b * 255)
			graphic.line(prev_x, prev_y, x, y)
		}
		prev_x = x
		prev_y = y
	}
}

function scale_point(x) {
	return x * (graphics_size - border_weight * 2) + border_weight
}
