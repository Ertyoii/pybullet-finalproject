from train import *
from model import *

PATH = './'


def test_load_model():
    net = NaiveNet()
    net.load_state_dict(torch.load(PATH))

    input, label = load_data()

    first_input = torch.from_numpy(input[0].reshape(1, 1, 256, 256)).float()
    print(net(first_input))
    print(label[0])


def test():
    dtype = torch.float32
    x = torch.zeros((64, 1, 256, 256), dtype=dtype)
    y = torch.ones((64, 3), dtype=dtype)

    model = NaiveNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for _ in range(5):
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()

        print(outputs)

    torch.save(model.state_dict(), PATH)
