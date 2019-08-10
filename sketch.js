var train = true;
function setup(){
    createCanvas(500,500);
    background(0);

    rn = new RedeNeural(2,3,1);
    
    //XOR problem
    dataset = {
        inputs:
            [[1,1],
            [1, 0],
            [0, 1],
            [0, 0]],
        outputs:
            [[0],
            [1],
            [1],
            [0]]
        }
}

function draw(){
    if(train){
        for(var i = 0; i < 10000; i++){
            var index = floor(random(4));
            rn.train(dataset.inputs[index], dataset.outputs[index]);
        }

        if(rn.predict([0,0])[0] < 0.04 && rn.predict([1,0])[0] > 0.98 ){
            train = false;
            console.log("terminou");
        }
    }
}