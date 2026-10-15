import { test } from "bun:test";
import assert from "node:assert/strict";
import { z } from "zod";
import {
  createPiRuntime,
  PiModelSource,
  type PiModel,
  type PiRuntime,
} from "../src/model-sources/pi.ts";
import { ModelSourceError } from "../src/model-source.ts";

const model = {
  provider: "anthropic",
  id: "claude-sonnet",
  name: "Claude Sonnet",
  reasoning: true,
};

function source(runtime: PiRuntime = runtimeWithModel()): PiModelSource {
  return new PiModelSource(async () => ({
    runtime,
    profiles: (item) => (item.reasoning ? ["off", "low", "high"] : []),
  }));
}

function runtimeWithModel(): PiRuntime {
  return {
    getAvailable: async (provider) => (!provider || provider === model.provider ? [model] : []),
    getModel: (provider, id) =>
      provider === model.provider && id === model.id ? model : undefined,
    completeSimple: async () => ({
      content: [{ type: "text", text: '{"tab":"Repair Socket Reconnect","reason":"task"}' }],
      stopReason: "stop",
    }),
  };
}

test("Pi exposes only authenticated providers and their staged model profiles", async () => {
  const adapter = source();
  assert.deepEqual(await adapter.listProviders(), [
    { id: "anthropic", label: "anthropic" },
  ]);
  assert.deepEqual(await adapter.listModels("anthropic"), [
    { id: "claude-sonnet", label: "Claude Sonnet" },
  ]);
  assert.deepEqual(await adapter.listProfiles("anthropic", "claude-sonnet"), [
    { id: "off", label: "off" },
    { id: "low", label: "low" },
    { id: "high", label: "high" },
  ]);
  await assert.rejects(adapter.listModels("openai"), /Pi provider is not connected/);
});

test("Pi completion uses Pi-owned model and thinking configuration", async () => {
  const calls: Array<{
    model: string;
    context: Parameters<PiRuntime["completeSimple"]>[1];
    options: Parameters<PiRuntime["completeSimple"]>[2];
  }> = [];

  const adapter = source({
    ...runtimeWithModel(),
    completeSimple: async (selected, context, options) => {
      calls.push({ model: selected.id, context, options });

      return {
        content: [{ type: "text", text: '{"tab":"Repair Socket Reconnect","reason":"task"}' }],
        stopReason: "stop",
      };
    },
  });

  const signal = AbortSignal.timeout(1_000);

  const text = await adapter.complete({
    selection: {
      version: 1,
      source: "pi",
      provider: "anthropic",
      model: "claude-sonnet",
      profile: "high",
    },
    context: { project: "Plugin", userRequests: ["Fix reconnect"] },
    system: "JSON only",
    prompt: "Name this task",
    maxOutputTokens: 123,
    maxRetries: 1,
    abortSignal: signal,
  });

  assert.match(text, /Repair Socket Reconnect/);
  const call = calls[0];

  assert.ok(call);
  assert.deepEqual(call.options, {
    reasoning: "high",
    maxTokens: 123,
    maxRetries: 1,
    signal,
  });
  const firstMessage = call.context.messages[0];

  assert.ok(firstMessage);
  assert.deepEqual(call.context, {
    systemPrompt: "JSON only",
    messages: [
      {
        role: "user",
        content: "Name this task",
        timestamp: firstMessage.timestamp,
      },
    ],
  });
});

test("Pi forwards a fresh OpenCode session id and omits it for other providers", async () => {
  const calls: Array<{ provider: string; sessionId: string | undefined; forwarded: boolean }> = [];

  const models: readonly PiModel[] = [
    { provider: "opencode", id: "fable", name: "Fable", reasoning: false, baseUrl: "https://example.com/models" },
    { provider: "opencode-go", id: "minimax", name: "MiniMax", reasoning: false, baseUrl: "https://opencode.ai/zen/go" },
    { provider: "custom", id: "hosted", name: "Hosted", reasoning: false, baseUrl: "https://opencode.ai/zen" },
    { provider: "anthropic", id: "claude-sonnet", name: "Claude Sonnet", reasoning: false, baseUrl: "https://api.anthropic.com" },
    { provider: "openai-codex", id: "gpt-5-codex", name: "Codex", reasoning: false, baseUrl: "https://chatgpt.com/backend-api/codex" },
    { provider: "openrouter", id: "router", name: "Router", reasoning: false, baseUrl: "https://openrouter.ai/api/v1" },
    { provider: "bait", id: "bait", name: "Bait", reasoning: false, baseUrl: "https://example.com/opencode.ai/v1" },
  ];

  // The spy is the engine under createPiRuntime. Dropping its sessionId copy fails this test.
  const resources = await createPiRuntime({
    getAvailable: async (provider) => models.filter((item) => !provider || item.provider === provider),
    getModel: (provider, id) => models.find((item) => item.provider === provider && item.id === id),
    completeSimple: async (model, _context, options) => {
      calls.push({
        provider: model.provider,
        sessionId: options.sessionId,
        forwarded: Object.hasOwn(options, "sessionId"),
      });

      return {
        content: [{ type: "text", text: '{"tab":"Repair Socket Reconnect","reason":"task"}' }],
        stopReason: "stop",
      };
    },
  });

  const adapter = new PiModelSource(async () => resources);

  const request = {
    context: { project: "Plugin", userRequests: ["Fix reconnect"] },
    system: "JSON only",
    prompt: "Name this task",
    maxOutputTokens: 123,
    maxRetries: 1,
    abortSignal: AbortSignal.timeout(1_000),
  };

  await adapter.complete({ ...request, selection: { version: 1, source: "pi", provider: "opencode", model: "fable" } });
  await adapter.complete({ ...request, selection: { version: 1, source: "pi", provider: "opencode", model: "fable" } });
  await adapter.complete({ ...request, selection: { version: 1, source: "pi", provider: "opencode-go", model: "minimax" } });
  await adapter.complete({ ...request, selection: { version: 1, source: "pi", provider: "custom", model: "hosted" } });
  await adapter.complete({ ...request, selection: { version: 1, source: "pi", provider: "anthropic", model: "claude-sonnet" } });
  await adapter.complete({ ...request, selection: { version: 1, source: "pi", provider: "openai-codex", model: "gpt-5-codex" } });
  await adapter.complete({ ...request, selection: { version: 1, source: "pi", provider: "openrouter", model: "router" } });
  await adapter.complete({ ...request, selection: { version: 1, source: "pi", provider: "bait", model: "bait" } });

  const sessionIdPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  const openCodeCalls = calls.filter((call) => call.provider === "opencode");
  const firstOpenCode = openCodeCalls[0];
  const secondOpenCode = openCodeCalls[1];

  assert.ok(firstOpenCode?.sessionId);
  assert.ok(secondOpenCode?.sessionId);
  assert.match(firstOpenCode.sessionId, sessionIdPattern);
  assert.match(secondOpenCode.sessionId, sessionIdPattern);
  assert.notEqual(firstOpenCode.sessionId, secondOpenCode.sessionId);
  assert.equal(firstOpenCode.forwarded, true);
  assert.equal(secondOpenCode.forwarded, true);

  for (const provider of ["opencode-go", "custom"]) {
    const call = calls.find((item) => item.provider === provider);

    assert.ok(call?.sessionId);
    assert.match(call.sessionId, sessionIdPattern);
    assert.equal(call.forwarded, true);
  }

  for (const provider of ["anthropic", "openai-codex", "openrouter", "bait"]) {
    const call = calls.find((item) => item.provider === provider);

    assert.ok(call);
    assert.equal(call.sessionId, undefined);
    assert.equal(call.forwarded, false);
  }
});

test("Pi failures are fail-closed and secret-free", async () => {
  const adapter = source({
    getAvailable: async () => [],
    getModel: () => model,
    completeSimple: async () => {
      throw new Error("Bearer private-token");
    },
  });

  const failure = await adapter.complete({
      selection: {
        version: 1,
        source: "pi",
        provider: "anthropic",
        model: "claude-sonnet",
      },
      context: { project: "Plugin", userRequests: ["Fix reconnect"] },
      system: "JSON only",
      prompt: "Name this task",
      maxOutputTokens: 123,
      maxRetries: 1,
      abortSignal: AbortSignal.timeout(1_000),
    }).then(
      () => { throw new Error("Expected Pi to reject an unavailable provider"); },
      (cause) => z.instanceof(ModelSourceError).parse(cause),
    );

  assert.match(failure.message, /not connected/);
  assert.doesNotMatch(failure.message, /private-token/);
});
