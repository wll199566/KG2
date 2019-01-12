import torch

def save(filename, epoch, model1, model2, optimizer, scheduler = None):
    state = {
        'epoch': epoch,
        'model1_state_dict': model1.state_dict(),
        'model2_state_dict': model2.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
        # ...
    }
    torch.save(state, filename)

def load(filename, model1, model2, optimizer, epoch, scheduler = None):
    
    state = torch.load(filename)

    model1.load_state_dict(state['model1_state_dict'])
    model2.load_state_dict(state['model2_state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    epoch = state['epoch'] + 1
    scheduler.load_state_dict(state['scheduler'])

    return epoch, model1, model2, optimizer, scheduler

def save_model(model1, model2, filename):

    state = {
        'model1_state_dict': model1.state_dict(),
        'model2_state_dict': model2.state_dict()
    }
    torch.save(state, filename)

def load_model(model1, model2, filename):
    
    state = torch.load(filename)
    model1.load_state_dict(state['model1_state_dict'])
    model2.load_state_dict(state['model2_state_dict'])

    return model1, model2

#with no code...
#def save_model(filename, model):
#    torch.save(model, filename)

#with no code...
#def load_model(filename):
#    model = torch.load(filename)
#    return model