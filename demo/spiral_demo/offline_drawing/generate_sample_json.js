var fs = require('fs')

function sampleActions(length) {
	// create a random action sequence.
	var res = []

	for (var i = 0; i < length; i++) {
		var x = Math.random()
		var y = Math.random()
		var p = 1.0
		var r = Math.random()
		var g = Math.random()
		var b = Math.random()
		var q = Math.floor(Math.random() * 2)
		res.push([x, y, p, r, g, b, q])
	}

	return res
}

var data = {
    'canvas_size': [100, 100],
    'version': '0.1',
    'actions': sampleActions(100)
}

let stringdata = JSON.stringify(data)
fs.writeFile('foo.json', JSON.stringify(data), (err) => {
    if(err) throw err;
    console.log('Data written to a file')
})