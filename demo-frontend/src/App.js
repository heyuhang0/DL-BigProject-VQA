import React from 'react';
import axios from 'axios';
import { useEffect, useState, useRef } from 'react';
import { Input, Upload, message, Select, Row, Col } from 'antd';
import { FileImageOutlined } from '@ant-design/icons';
import { MessageList } from 'react-chat-elements';

import './App.css';
import 'react-chat-elements/dist/main.css';


const { Search } = Input;
const { Option } = Select;

function getBase64(img, callback) {
  const reader = new FileReader();
  reader.addEventListener('load', () => callback(reader.result));
  reader.readAsDataURL(img);
}

function App() {
  // message list
  const [messages, setMessages] = useState([{
    position: 'left',
    type: 'text',
    text: 'Upload an image and ask questions',
    date: new Date(),
  }]);
  const appendMessages = (message) => {
    setMessages(prevMessages => prevMessages.concat([message]))
  }

  // scroll to bottom on new message
  const messagesBottomRef = useRef(null);
  useEffect(() => {
    messagesBottomRef.current.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // VQA states
  const [image, setImage] = useState(null);
  const [question, setQuestion] = useState('');
  const defaultModel = 'attention_all';
  const [model, setModel] = useState(defaultModel);

  return (
    <div className="App">
      <header>
        <Row>
          <Col span={12}>
            <h1>VQA Demo</h1>
          </Col>
          <Col span={12}>
            <Select className="model-selector" defaultValue={defaultModel}onChange={setModel}>
              <Option value="attention_all">Attention(single word)</Option>
              {/* <Option value="attention_y/n">Attention(yes/no)</Option> */}
            </Select>
          </Col>
        </Row>
      </header>
      <main>
        <div className="chat-messages">
          <MessageList
            toBottomHeight="100%"
            dataSource={messages}
          />
          <div ref={messagesBottomRef} />
        </div>
      </main>
      <footer>
        <Search
          className="chat-input"
          placeholder="Ask a question"
          enterButton="Send"
          size="large"
          value={question}
          suffix={
            <Upload
              beforeUpload={file => {
                const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';
                if (!isJpgOrPng) {
                  message.error('You can only upload JPG/PNG file!');
                  return false;
                }
                getBase64(file, base64Url => {
                  setImage(base64Url);
                  appendMessages({
                    position: 'right',
                    type: 'photo',
                    data: {
                      uri: base64Url
                    },
                    date: new Date(),
                  });
                });
                return false;
              }}
              fileList={[]}
            >
              <FileImageOutlined style={{ fontSize: 24 }} />
            </Upload>
          }
          onChange={e => setQuestion(e.target.value)}
          onSearch={m => {
            if (m === '') {
              return;
            }
            setQuestion('');
            appendMessages({
              position: 'right',
              type: 'text',
              text: m,
              date: new Date(),
            });
            if (image === null) {
              appendMessages({
                position: 'left',
                type: 'text',
                text: 'Please upload an image first :)',
                date: new Date(),
              });
            } else {
              axios.post('/api/vqa', {
                image: image,
                question: question,
                model: model,
              }).then(res => {
                res.data.answer.forEach(msg => {
                  if (msg.startsWith('data:image')) {
                    appendMessages({
                      position: 'left',
                      type: 'photo',
                      data: {
                        uri: msg
                      },
                      date: new Date(),
                    });
                  } else {
                    appendMessages({
                      position: 'left',
                      type: 'text',
                      text: msg,
                      date: new Date(),
                    });
                  }
                });
              }).catch(e => {
                console.log(e);
                let message = 'unknown backend error';
                if (e.response.data.message) {
                  message = e.response.data.message;
                }
                appendMessages({
                  position: 'left',
                  type: 'text',
                  text: message,
                  date: new Date(),
                });
              });
            }
          }}
        />
      </footer>
    </div>
  );
}

export default App;
