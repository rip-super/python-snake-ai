# Python Snake AI

run `python agent.py` to start training. 

`model_save.pt` is a model that i spent a while training, so it is decently good at the game. 
if you dont want to start the training from scratch and instead use the pretrained model, just edit the code in agent.py from this:
```python
if __name__ == "__main__":
    train()
    #train(pretrained_model="model_save.pt")
```

to this:
```python
if __name__ == "__main__":
    #train()
    train(pretrained_model="model_save.pt")
```

Also, if you wish play the snake game (mabye to challenge the ai) just run 
```bash
python game_human.py
```
