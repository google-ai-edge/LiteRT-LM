/**
 * Copyright 2026 The ODML Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {EngineFake} from '@litert-lm/core/testing/fakes';

import {Message, Tool, ToolResponsePart} from '../conversation_config.js';
import {Engine} from '../engine.js';
import {JsonObject, JsonValue} from '../types.js';

import {AutoToolChat, ToolProgressEvent, ToolWithImplementation} from './auto_tool_chat.js';

describe('AutoToolChat', () => {
  let engine: EngineFake&Engine;

  beforeEach(() => {
    engine =
        new EngineFake({model: 'fake_model'}) as unknown as EngineFake & Engine;
  });

  it('throws if tools are provided via ConversationConfig.preface',
     async () => {
       const tools = [{name: 'dummy_tool', execute: async () => ({})}];

       expect(() => {
         return new AutoToolChat({
           engine,
           config: {
             preface: {tools: [{name: 'dummy_tool', execute: async () => ({})}]}
           }
         });
       })
           .toThrowError(
               /Do not provide tools via ConversationConfig.preface.tools/);
     });

  it('calls a tool and loops back into the model (sendMessage)', async () => {
    const executeSpy = jasmine.createSpy('execute').and.returnValue(
        Promise.resolve({temperature: 72, weather: 'sunny'}));
    const tools: ToolWithImplementation[] = [{
      name: 'get_current_weather',
      description: 'Gets current weather in a location',
      parameters: {
        type: 'object',
        properties: {location: {type: 'string'}},
        required: ['location']
      },
      execute: async (args: Record<string, JsonValue>) => executeSpy(args)
    }];

    const conversation = new AutoToolChat({engine, tools});

    engine.cachedConversation.queueResponse({
      role: 'assistant',
      tool_calls: [{
        type: 'function',
        function:
            {name: 'get_current_weather', arguments: {location: 'Seattle'}}
      }]
    });
    engine.cachedConversation.queueResponse({
      role: 'assistant',
      content: 'The weather in Seattle is sunny and 72 degrees.'
    });

    const response = await conversation.sendMessage(
        {role: 'user', content: 'What is the current weather in Seattle?'});

    expect(response).toBeDefined();
    expect(response.content)
        .toBe('The weather in Seattle is sunny and 72 degrees.');
    expect(executeSpy).toHaveBeenCalledOnceWith(jasmine.objectContaining({
      location: 'Seattle'
    }));
  });

  it('calls a tool and streams the response (sendMessageStreaming)',
     async () => {
       const executeSpy = jasmine.createSpy('execute').and.returnValue(
           Promise.resolve({temperature: 72, weather: 'sunny'}));
       const tools = [{
         name: 'get_current_weather',
         description: 'Gets current weather in a location',
         execute: async (args: Record<string, JsonValue>) => executeSpy(args)
       }];

       const conversation = new AutoToolChat({engine, tools});

       engine.cachedConversation.queueResponse({
         role: 'assistant',
         tool_calls: [{
           type: 'function',
           function:
               {name: 'get_current_weather', arguments: {location: 'Seattle'}}
         }]
       });
       engine.cachedConversation.queueResponse({
         role: 'assistant',
         content: 'The weather in Seattle is sunny and 72 degrees.'
       });

       const stream = conversation.sendMessageStreaming(
           {role: 'user', content: 'What is the current weather in Seattle?'});

       const reader = stream.getReader();
       let resultCount = 0;
       while (true) {
         const {done} = await reader.read();
         if (done) break;
         resultCount++;
       }

       // We should only receive the second response, since the first had only a
       // tool_call, which is now filtered out.
       expect(resultCount).toBe(1);
       expect(executeSpy).toHaveBeenCalledOnceWith(jasmine.objectContaining({
         location: 'Seattle'
       }));
     });

  it('stops and throws an error if recursion limit is exceeded', async () => {
       const tools: ToolWithImplementation[] = [{
         name: 'loop_tool',
         description: 'Keeps looping',
         execute: async () => ({again: true})
       }];

       const conversation =
           new AutoToolChat({engine, tools, recurringToolCallLimit: 3});

       // Enqueue 5 of them.
       for (let i = 0; i < 5; i++) {
         engine.cachedConversation.queueResponse({
           role: 'assistant',
           tool_calls: [
             {type: 'function', function: {name: 'loop_tool', arguments: {}}}
           ]
         });
       }

       await expectAsync(conversation.sendMessage('start'))
           .toBeRejectedWithError(/Tool calling exceeded the recurring limit/);
     });

  it('throws an error if a tool is provided without an execute implementation',
     () => {
       const tools = [{
         name: 'unimplemented_tool',
       } as Tool];

       expect(() => {
         return new AutoToolChat(
             {engine, tools: tools as ToolWithImplementation[]});
       }).toThrowError(/Tool unimplemented_tool must have an execute function/);
     });

  it('throws an error if duplicate tool names are provided', () => {
    const tools = [
      {
        name: 'duplicate_tool',
        execute: async () => ({}),
      },
      {
        name: 'duplicate_tool',
        execute: async () => ({}),
      }
    ];

    expect(() => {
      return new AutoToolChat({engine, tools});
    }).toThrowError(/Duplicate tool name duplicate_tool provided/);
  });

  it('notifies the model if a call specifies an unknown tool (sendMessage)',
     async () => {
       const tools = [{name: 'dummy_tool', execute: async () => ({})}];
       const conversation = new AutoToolChat({engine, tools});

       engine.cachedConversation.queueResponse({
         role: 'assistant',
         tool_calls: [
           {type: 'function', function: {name: 'unknown_tool', arguments: {}}}
         ]
       });
       engine.cachedConversation.queueResponse(
           {role: 'assistant', content: 'Oops, that tool doesnt exist.'});

       const response = await conversation.sendMessage('call unknown tool');

       expect(response.content).toBe('Oops, that tool doesnt exist.');
       // Verify the fake tool response was enqueued internally.
       // Because it returns the second response we know it looped back.
       // We can also examine the faked history to ensure the error was
       // injected.
       const history = engine.cachedConversation.getHistory();
       const userMsg =
           history[1];  // [user input, model output, user response...] ->
                        // actually: history[0] = user input, history[1] = model
                        // tool call, history[2] = fake tool_response
       expect(history.length).toBe(4);
       expect((userMsg as Message).role).toEqual('assistant');
       expect((history[2] as Message).role).toEqual('tool');
       const content = (history[2] as Message).content as ToolResponsePart[];
       expect((content[0].response as JsonObject)['error'] as string)
           .toContain('Tool unknown_tool not found');
     });

  it('notifies the model if a call specifies an unknown tool (sendMessageStreaming)',
     async () => {
       const tools = [{name: 'dummy_tool', execute: async () => ({})}];
       const conversation = new AutoToolChat({engine, tools});

       engine.cachedConversation.queueResponse({
         role: 'assistant',
         tool_calls: [
           {type: 'function', function: {name: 'unknown_tool', arguments: {}}}
         ]
       });
       engine.cachedConversation.queueResponse(
           {role: 'assistant', content: 'Oops, that streaming tool doesnt exist.'});

       const stream = conversation.sendMessageStreaming('call unknown tool');
       const reader = stream.getReader();

       const chunks: Message[] = [];
       while (true) {
         const {done, value} = await reader.read();
         if (done) break;
         chunks.push(value);
       }

       // We should only receive the second response, since the first had only a
       // tool_call
       expect(chunks.length).toBe(1);
       expect(chunks[0].content)
           .toBe('Oops, that streaming tool doesnt exist.');

       const history = engine.cachedConversation.getHistory();
       expect((history[2] as Message).role).toEqual('tool');
       const content = (history[2] as Message).content as ToolResponsePart[];
       expect((content[0].response as JsonObject)['error'] as string)
           .toContain('Tool unknown_tool not found');
     });

  describe('Parallel and Error Handling', () => {
    it('surfaces an error back to the model if a tool throws', async () => {
      const executeSpy = jasmine.createSpy('execute').and.throwError(
          new Error('Internal Boom!'));
      const tools: ToolWithImplementation[] = [{
        name: 'error_tool',
        description: 'Throws an error',
        execute: async (args: Record<string, JsonValue>) => executeSpy(args)
      }];
      const conversation = new AutoToolChat({engine, tools});

      engine.cachedConversation.queueResponse({
        role: 'assistant',
        tool_calls:
            [{type: 'function', function: {name: 'error_tool', arguments: {}}}]
      });
      engine.cachedConversation.queueResponse(
          {role: 'assistant', content: 'Handled the error gracefully.'});

      const response = await conversation.sendMessage('call error tool');
      expect(response.content).toBe('Handled the error gracefully.');

      const history = engine.cachedConversation.getHistory();
      const content = (history[2] as Message).content as ToolResponsePart[];
      expect((content[0].response as JsonObject)['error'] as string)
          .toContain('Error executing tool error_tool: Error: Internal Boom!');
    });

    it('calls multiple tools simultaneously', async () => {
      let isFirstToolDone = false;
      let isSecondToolDone = false;
      let firstToolStarted = false;
      let secondToolStarted = false;

      let unlockFirstTool: () => void;
      const firstToolPromise = new Promise<void>((resolve) => {
        unlockFirstTool = resolve;
      });

      const tools = [
        {
          name: 'tool_one',
          execute: async () => {
            firstToolStarted = true;
            await firstToolPromise;
            isFirstToolDone = true;
            return {data: 'one'};
          }
        },
        {
          name: 'tool_two',
          execute: async () => {
            secondToolStarted = true;
            // The fact that this evaluates means both started before tool_one
            // finished.
            expect(firstToolStarted).toBeTrue();
            unlockFirstTool!();
            isSecondToolDone = true;
            return {data: 'two'};
          }
        }
      ];

      const conversation = new AutoToolChat({engine, tools});

      engine.cachedConversation.queueResponse({
        role: 'assistant',
        tool_calls: [
          {type: 'function', function: {name: 'tool_one', arguments: {}}},
          {type: 'function', function: {name: 'tool_two', arguments: {}}}
        ]
      });
      engine.cachedConversation.queueResponse(
          {role: 'assistant', content: 'Both done.'});

      const response = await conversation.sendMessage('call both tools');
      expect(response.content).toBe('Both done.');
      expect(isFirstToolDone).toBeTrue();
      expect(isSecondToolDone).toBeTrue();
      expect(secondToolStarted).toBeTrue();
    });

    it('handles simultaneous tool calls where one fails', async () => {
      const tools = [
        {
          name: 'tool_success',
          execute: async () => {
            return {data: 'success'};
          }
        },
        {
          name: 'tool_fail',
          execute: async () => {
            throw new Error('Boom!');
          }
        }
      ];

      const conversation = new AutoToolChat({engine, tools});

      engine.cachedConversation.queueResponse({
        role: 'assistant',
        tool_calls: [
          {type: 'function', function: {name: 'tool_success', arguments: {}}},
          {type: 'function', function: {name: 'tool_fail', arguments: {}}}
        ]
      });
      engine.cachedConversation.queueResponse(
          {role: 'assistant', content: 'Both handled.'});

      const response = await conversation.sendMessage('call both tools');
      expect(response.content).toBe('Both handled.');

      const history = engine.cachedConversation.getHistory();
      const content = (history[2] as Message).content as ToolResponsePart[];

      expect(content[0].name).toBe('tool_success');
      expect(String((content[0].response as JsonObject)['data']))
          .toBe('success');

      expect(content[1].name).toBe('tool_fail');
      expect((content[1].response as JsonObject)['error'] as string)
          .toContain('Error executing tool tool_fail: Error: Boom!');
    });

    it('handles simultaneous tool calls where one does not exist', async () => {
      const tools = [{
        name: 'tool_success',
        execute: async () => {
          return {data: 'success'};
        }
      }];

      const conversation = new AutoToolChat({engine, tools});

      engine.cachedConversation.queueResponse({
        role: 'assistant',
        tool_calls: [
          {type: 'function', function: {name: 'tool_success', arguments: {}}},
          {type: 'function', function: {name: 'tool_not_exist', arguments: {}}}
        ]
      });
      engine.cachedConversation.queueResponse(
          {role: 'assistant', content: 'Both handled.'});

      const response = await conversation.sendMessage('call both tools');
      expect(response.content).toBe('Both handled.');

      const history = engine.cachedConversation.getHistory();
      const content = (history[2] as Message).content as ToolResponsePart[];

      expect(content[0].name).toBe('tool_success');
      expect(String((content[0].response as JsonObject)['data']))
          .toBe('success');

      expect(content[1].name).toBe('tool_not_exist');
      expect((content[1].response as JsonObject)['error'] as string)
          .toContain('Tool tool_not_exist not found');
    });

    describe('ToolProgressEvent', () => {
      it('fires started and completed events over a successful tool call',
         async () => {
           let executeCalled = false;
           const tools = [{
             name: 'test_tool',
             execute: async () => {
               executeCalled = true;
               return {outcome: 'ok'};
             }
           }];

           const events: ToolProgressEvent[] = [];
           const conversation = new AutoToolChat(
               {engine, tools, onToolProgress: (e) => events.push(e)});

           engine.cachedConversation.queueResponse({
             role: 'assistant',
             tool_calls: [{
               type: 'function',
               function: {name: 'test_tool', arguments: {param: 1}}
             }]
           });
           engine.cachedConversation.queueResponse(
               {role: 'assistant', content: 'Finished.'});

           await conversation.sendMessage('start');

           expect(executeCalled).toBeTrue();
           expect(events.length).toBe(2);

           expect(events[0]).toEqual(jasmine.objectContaining({
             id: jasmine.any(Number),
             name: 'test_tool',
             arguments: {param: 1},
             status: 'started'
           }));

           expect(events[1]).toEqual(jasmine.objectContaining({
             id: events[0].id,
             name: 'test_tool',
             arguments: {param: 1},
             status: 'completed',
             result: {outcome: 'ok'}
           }));
         });

      it('fires started and error events over a failed execute tool call',
         async () => {
           const tools = [{
             name: 'error_tool',
             execute: async () => {
               throw new Error('Boom!');
             }
           }];

           const events: ToolProgressEvent[] = [];
           const conversation = new AutoToolChat(
               {engine, tools, onToolProgress: (e) => events.push(e)});

           engine.cachedConversation.queueResponse({
             role: 'assistant',
             tool_calls: [
               {type: 'function', function: {name: 'error_tool', arguments: {}}}
             ]
           });
           engine.cachedConversation.queueResponse(
               {role: 'assistant', content: 'Finished.'});

           await conversation.sendMessage('start');

           expect(events.length).toBe(2);

           expect(events[0]).toEqual(jasmine.objectContaining({
             id: jasmine.any(Number),
             name: 'error_tool',
             status: 'started'
           }));

           expect(events[1]).toEqual(jasmine.objectContaining({
             id: events[0].id,
             name: 'error_tool',
             status: 'error',
             error: 'Error: Boom!'
           }));
           expect(events[1].result).toBeUndefined();
         });

      it('fires started and error events when a requested tool does not exist',
         async () => {
           const events: ToolProgressEvent[] = [];
           const conversation = new AutoToolChat(
               {engine, tools: [], onToolProgress: (e) => events.push(e)});

           engine.cachedConversation.queueResponse({
             role: 'assistant',
             tool_calls: [
               {type: 'function', function: {name: 'fake_tool', arguments: {}}}
             ]
           });
           engine.cachedConversation.queueResponse(
               {role: 'assistant', content: 'Finished.'});

           await conversation.sendMessage('start');

           expect(events.length).toBe(2);

           expect(events[0]).toEqual(jasmine.objectContaining({
             id: jasmine.any(Number),
             name: 'fake_tool',
             status: 'started'
           }));

           expect(events[1]).toEqual(jasmine.objectContaining({
             id: events[0].id,
             name: 'fake_tool',
             status: 'error',
             error: 'Tool fake_tool not found.'
           }));
         });

      it('issues unique incrementing ids per tool call even within the same iteration',
         async () => {
           const tools = [
             {name: 'toolA', execute: async () => ({})},
             {name: 'toolB', execute: async () => ({})}
           ] as ToolWithImplementation[];

           const events: ToolProgressEvent[] = [];
           const conversation = new AutoToolChat(
               {engine, tools, onToolProgress: (e) => events.push(e)});

           engine.cachedConversation.queueResponse({
             role: 'assistant',
             tool_calls: [
               {type: 'function', function: {name: 'toolA', arguments: {}}},
               {type: 'function', function: {name: 'toolB', arguments: {}}}
             ]
           });
           engine.cachedConversation.queueResponse(
               {role: 'assistant', content: 'Finished.'});

           await conversation.sendMessage('start');

           // We expect 4 events: A started, B started, A completed, B completed
           // (though completion order might vary)
           expect(events.length).toBe(4);

           const startedEvents = events.filter(e => e.status === 'started');
           expect(startedEvents.length).toBe(2);

           expect(startedEvents[0].id).not.toBe(startedEvents[1].id);
           expect(startedEvents[0].name).not.toBe(startedEvents[1].name);
         });

      it('ignores tool execution completions if cancelled while waiting',
         async () => {
           let resolveTool: (value: unknown) => void;
           const toolPromise = new Promise((r) => resolveTool = r);

           const tools = [{
             name: 'long_tool',
             execute: async () => {
               await toolPromise;
               return {data: 'ok'};
             }
           }];

           const events: ToolProgressEvent[] = [];
           const conversation = new AutoToolChat(
               {engine, tools, onToolProgress: (e) => events.push(e)});

           engine.cachedConversation.queueResponse({
             role: 'assistant',
             tool_calls: [
               {type: 'function', function: {name: 'long_tool', arguments: {}}}
             ]
           });
           engine.cachedConversation.queueResponse(
               {role: 'assistant', content: 'Finished.'});

           // Fire it off, but don't await because it will get stuck waiting for
           // `toolPromise`
           const sendPromise = conversation.sendMessage('start');

           // Give it a tick to evaluate queueResponse and invoke the tool
           await new Promise(r => setTimeout(r, 10));

           expect(events.length).toBe(1);
           expect(events[0].status).toBe('started');

           // Cancel the conversation
           conversation.cancel();

           // Now resolve the tool
           resolveTool!({data: 'ok'});

           // Wait for the sendPromise to reject since we cancelled during
           // execution
           await expectAsync(sendPromise)
               .toBeRejectedWithError('Conversation cancelled');

           // Verify NO completion event was fired
           expect(events.length).toBe(1);
         });
    });
  });
});
