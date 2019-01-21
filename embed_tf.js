let width = d3.select("body").style("width");
let height = d3.select("body").style("height");
let svg = d3.select("svg").attr('width', width).attr('height', height);
d3.csv("/data.csv", function(r){
    row = []
    for( i = 0 ; i < Object.keys(r).length;i++){
        val = parseFloat(r[i])
        row.push(val)
    }
    return row
}).then(
   function(data){
       main(data)
    }
)
function main(data){
    data = tf.tensor2d(data);
    console.log(data.shape);
    X = data.slice([0, 0], [data.shape[0], data.shape[1] - 1]);
    y = data.slice([0, data.shape[1] - 1], [data.shape[0], 1]).reshape([data.shape[0]]);
    console.log(X.shape, y.shape)
    const model = tf.sequential();
    input_size = X.shape[1]
    hidden = 64
    act = 'selu'
    model.add(tf.layers.dense({units: hidden, inputShape: [input_size], 'activation': act}));
    model.add(tf.layers.dense({units: hidden, 'activation': act}))
    model.add(tf.layers.dense({units: 2}))
    model.add(tf.layers.dense({units: hidden, 'activation': act}))
    model.add(tf.layers.dense({units: hidden, 'activation': act}))
    model.add(tf.layers.dense({units: input_size}));
    const embed = tf.model({inputs: model.inputs, outputs: model.layers[2].output});
    opt = tf.train.adam(0.001)
    model.compile({loss: 'meanSquaredError', optimizer: opt});
    opts = {
        epochs: 1,
        batchSize: 128,
        callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    const loss = model.evaluate(X, X);
                    loss.print()
                }
        }
    }
    H = embed.predict(X)
    let svg = d3.select("svg").attr('width', width).attr('height', height);
    min = H.min(0).dataSync()
    max = H.max(0).dataSync()
    let xscale = d3.scaleLinear().domain([min[0], max[0]]).range([0, width]);
    let yscale = d3.scaleLinear().domain([min[1], max[1]]).range([0, height]);
    //let color = d3.interpolateSpectral;
    let color = d3.scaleOrdinal(d3.schemeCategory10);

    svg.selectAll('circle')
        .data(d3.range(H.shape[0] - 1)) 
        .enter()
        .append('circle')
        .attr('cx', function(d){return xscale(H.get(d, 0))})
        .attr('cy', function(d){return yscale(H.get(d, 1))})
        .attr('r',  6)
        .attr('fill', function(d, i){return color(y.get(d))})
        .append("svg:title")
          .text(function(d, i) { return y.get(d)});
    epochs = 0
    nb_epochs = 1000
    function fit(){
        epochs += 1
        if(epochs == nb_epochs){
            return;
        }
        model.fit(X, X, opts).then(() => {
            H = embed.predict(X)
            let svg = d3.select("svg").attr('width', width).attr('height', height);
            min = H.min(0).dataSync()
            max = H.max(0).dataSync()
            let xscale = d3.scaleLinear().domain([min[0], max[0]]).range([0, width]);
            let yscale = d3.scaleLinear().domain([min[1], max[1]]).range([0, height]);
            svg.selectAll('circle')
                .data(d3.range(H.shape[0] - 1))
                .transition()
                .duration(100)
                .attr('cx', function(d){return xscale(H.get(d, 0))})
                .attr('cy', function(d){return yscale(H.get(d, 1))})
                .attr('r',  6)
                .attr('fill', function(d, i){return color(y.get(d))})
            fit();
        });
    }
    fit()
}
