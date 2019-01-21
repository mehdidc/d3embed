let width = d3.select("body").style("width");
let height = d3.select("body").style("height");
let xscale = d3.scaleLinear().domain([-1, 1]).range([0, width]);
let yscale = d3.scaleLinear().domain([-1, 1]).range([0, height]);
function normal() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}
let nj_normal = function(shape){
    u = nj.random(shape);
    v = nj.random(shape);
    u = u.log().multiply(-2).sqrt()
    v = nj.cos(v.multiply(2*Math.PI))
    return u.multiply(v)
}
let generateRandomData = function(nb){
    let data = []
    for (let i = 0; i < nb; i++) {
        x = Math.random() * 2 - 1;
        y = x + normal() * 0.2;
        data.push([x, y]);
    }
    return data;
}
let data = generateRandomData(100);


let svg = d3.select("svg").attr('width', width).attr('height', height);
let circle = svg.selectAll('circle').data(data);
circle.enter().append('circle')
	.attr('cx', function(d){ return xscale(d[0])})
	.attr('cy', function(d){ return yscale(d[1])})
	.attr('r',  10)
        .attr('fill', function(d, i){return "red"})
let line = svg.append("line");
line.attr("stroke-width", 3).attr("stroke", "blue")

let X = nj.array(data);
let embed_size = 1;
let W = nj_normal([X.shape[1], embed_size]).multiply(0.01);
let DW_sqr = nj.zeros(W.shape);
let alpha = 0.01;
let batch_size = X.shape[0];
let one_epoch = function(W){
    for(i = 0; i < X.shape[0]; i+=batch_size){
        let start = i;
        let end = Math.min(i + batch_size, X.shape[0]);
        Xb = X.slice([start, end]);
        H = nj.dot(Xb, W);
        Y = nj.dot(H, W.T);
        L = Xb.subtract(Y).pow(2).sum() / Xb.shape[0];
        DY = Y.subtract(X).multiply(2).divide(Xb.shape[0])
        DH = nj.dot(DY, W);
        DW1 = nj.dot(Xb.T, DH)
        DW2 = nj.dot(DY.T, H)
        DW = DW1.add(DW2)
        // gradient with finite diff
        /*
        for(m = 0; m < W.shape[0];m++){
            for(n = 0; n < W.shape[1];n++){
                eps = 1e-7
                Wa = W.clone()
                Wa.set(m, n, W.get(m, n) - eps)
                
                Wb = W.clone()
                Wb.set(m, n, W.get(m, n) + eps) 
                
                H = nj.dot(Xb, Wa);
                Y = nj.dot(H, Wa.T);
                La = Xb.subtract(Y).pow(2).sum() / Xb.shape[0];

                H = nj.dot(Xb, Wb);
                Y = nj.dot(H, Wb.T);
                Lb = Xb.subtract(Y).pow(2).sum() / Xb.shape[0];

                DW_fd = (Lb - La) / (2*eps)
                delta = Math.abs(DW.get(m, n) - DW_fd)
                console.log(delta)
                //DW.set(m, n, DW_fd)
            }
        }
        */
        W = W.subtract(DW.multiply(alpha));
    }
    H = nj.dot(X, W);
    Y = nj.dot(H, W.T);
    L = X.subtract(Y).pow(2).sum();
    console.log(L);
    return W;
}

let cnt = 0;
let nb_epochs = 1000;
function run(){
    cnt += 1
    if(cnt == nb_epochs){
        return;
    }
    W = one_epoch(W);
    let W_sqr = Math.sqrt(W.pow(2).sum())
    let a = W.get(0, 0) / W_sqr;
    let b = W.get(1, 0) / W_sqr;
    let x1 = xscale(0);
    let y1 = yscale(0);
    let x2 = xscale(a);
    let y2= yscale(b);
    line.transition()
        .duration(10)
        .delay(0)
        .attr("x1", x1)
        .attr("y1", y1)
        .attr("x2", x2)
        .attr("y2", y2)
        .on("end", run)
}
run()
