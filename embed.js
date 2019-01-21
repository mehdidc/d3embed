let width = d3.select("body").style("width");
let height = d3.select("body").style("height");
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
        y = x + normal() * 0.01;
        data.push([x, y]);
    }
    return data;
}
let svg = d3.select("svg").attr('width', width).attr('height', height);
d3.csv("/data.csv", function(r){return [parseFloat(r.x), parseFloat(r.y), parseFloat(r.z)]}).then(
   function(data){
       //data = generateRandomData(100)
       main(data)
       //main(data)
    }
);
function main(data){
    let X = nj.array(data);
    console.log(X.shape)
    let embed_size = 2;
    let hidden_size = 32;
    let std1 = Math.sqrt(2/(X.shape[1] + hidden_size))
    let std2 = Math.sqrt(2/(hidden_size + embed_size))
    let std3 = Math.sqrt(2/(embed_size + X.shape[1]))
    let W1 = nj_normal([X.shape[1], hidden_size]).multiply(std1);
    let W2 = nj_normal([hidden_size, embed_size]).multiply(std2);
    let W3 = nj_normal([embed_size, X.shape[1]]).multiply(std3);

    let DW1_sqr = nj.zeros(W1.shape);
    let DW2_sqr = nj.zeros(W2.shape);
    let DW3_sqr = nj.zeros(W3.shape);

    let alpha = 0.01;
    let batch_size = 32;
    function add_constant(x){
        o = nj.ones(x.shape[0]).reshape(x.shape[0], 1)
        return nj.concatenate(x, o)
    }
    function forward(X, params){
        [W1, W2, W3, DW1_sqr, DW2_sqr, DW3_sqr] = params
        H1 = nj.dot(X, W1);
        H2 = nj.tanh(H1);
        H3 = nj.dot(H2, W2);
        Y = nj.dot(H3, W3);
        return [H1, H2, H3, Y]
    }
    function embed(X, params){
        [H1, H2, H3, Y] = forward(X, params)
        return H3;
    }
    function one_epoch(params){
        [W1, W2, W3, DW1_sqr, DW2_sqr, DW3_sqr] = params
        for(i = 0; i < X.shape[0]; i += batch_size){
            let start = i;
            let end = Math.min(i + batch_size, X.shape[0]);
            Xb = X.slice([start, end]);
            [H1, H2, H3, Y] = forward(Xb, params)
            DY = Y.subtract(Xb).multiply(2).divide(Xb.shape[0])
            DH3 = nj.dot(DY, W3.T);
            DH2 = nj.dot(DH3, W2.T)
            DH1 = nj.tanh(H1).pow(2).multiply(-1).add(1).multiply(DH2)
            DW1 = nj.dot(Xb.T, DH1)
            DW2 = nj.dot(H2.T, DH3)
            DW3 = nj.dot(H3.T, DY)
            /*
            // gradient check
            let p
            for(m = 0; m < W1.shape[0];m++){
                for(n = 0; n < W1.shape[1];n++){
                    eps = 1e-7
                    Wa = W1.clone()
                    Wa.set(m, n, W1.get(m, n) - eps)
                    Wb = W1.clone()
                    Wb.set(m, n, W1.get(m, n) + eps) 
                    p = [Wa, W2, W3, DW1_sqr, DW2_sqr, DW3_sqr];
                    [H1, H2, H3, Y] = forward(Xb, p)
                    La = Xb.subtract(Y).pow(2).sum() / Xb.shape[0];
                    p = [Wb, W2, W3, DW1_sqr, DW2_sqr, DW3_sqr];
                    [H1, H2, H3, Y] = forward(Xb, p)
                    Lb = Xb.subtract(Y).pow(2).sum() / Xb.shape[0];
                    DW_fd = (Lb - La) / (2*eps)
                    delta = Math.abs(DW1.get(m, n) - DW_fd)
                    console.log(delta)
                }
            }
            */
            DW1_sqr = DW1_sqr.add(DW1.pow(2))
            DW2_sqr = DW2_sqr.add(DW2.pow(2))
            DW3_sqr = DW3_sqr.add(DW3.pow(2))
            
            //let d1 = DW1.divide(nj.sqrt(DW1_sqr))
            //let d2 = DW2.divide(nj.sqrt(DW2_sqr))
            //let d3 = DW3.divide(nj.sqrt(DW3_sqr))
            let d1 = DW1
            let d2 = DW2
            let d3 = DW3
            W1 = W1.subtract(d1.multiply(alpha));
            W2 = W2.subtract(d2.multiply(alpha));
            W3 = W3.subtract(d3.multiply(alpha));
        }
        params = [W1, W2, W3, DW1_sqr, DW2_sqr, DW3_sqr]
        [H1, H2, H3, Y] = forward(X, [W1, W2, W3, DW1_sqr, DW2_sqr, DW3_sqr])
        L = X.subtract(Y).pow(2).sum();
        console.log(L)
        return [W1, W2, W3, DW1_sqr, DW2_sqr, DW3_sqr]

    }
    params = [W1, W2, W3, DW1_sqr, DW2_sqr, DW3_sqr]
    H = embed(X, params)
    let xscale = d3.scaleLinear().domain([H.min(), H.max()]).range([0, width]);
    let yscale = d3.scaleLinear().domain([H.min(), H.max()]).range([0, height]);
    svg.selectAll('circle')
        .data(d3.range(H.shape[0] - 1)) 
        .enter()
        .append('circle')
        .attr('cx', function(d){return xscale(H.get(d, 0))})
        .attr('cy', function(d){return yscale(H.get(d, 1))})
        .attr('r',  10)
        .attr('fill', function(d, i){return "red"})
    let cnt = 0;
    let nb_epochs = 1000; 
    function run(){
        for(k = 0;k < nb_epochs;k++){
            params = one_epoch(params)
        }
        H = embed(X, params)
        let xscale = d3.scaleLinear().domain([H.min(), H.max()]).range([0, width]);
        let yscale = d3.scaleLinear().domain([H.min(), H.max()]).range([0, height]);
        svg.selectAll('circle')
            .data(d3.range(H.shape[0] - 1))
            .attr('cx', function(d){return xscale(H.get(d, 0))})
            .attr('cy', function(d){return yscale(H.get(d, 1))})
    }
    function animated_run(){
        if(cnt == nb_epochs){
            return;
        }
        cnt += 1
        for(k = 0;k < nb_epochs;k++){
            params = one_epoch(params)
        }
        H =  embed(X, params)
        let xscale = d3.scaleLinear().domain([H.min(), H.max()]).range([0, width]);
        let yscale = d3.scaleLinear().domain([H.min(), H.max()]).range([0, height]);
        svg.selectAll('circle')
            .data(d3.range(H.shape[0] - 1))
            //.transition()
            //.duration(100)
            //.delay(0)
            .attr('cx', function(d){return xscale(H.get(d, 0))})
            .attr('cy', function(d){return yscale(H.get(d, 1))})
            //.on("end", run)
    }
    run()
}
