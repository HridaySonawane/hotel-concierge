import { spawn } from 'node:child_process';
import { NextResponse } from 'next/server';

export const runtime = 'nodejs';

type RequestBody = { guestId?: string; text?: string };

export async function POST(request: Request) {
  let body: RequestBody;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Request body must be valid JSON.' }, { status: 400 });
  }

  const guestId = body.guestId?.trim();
  const text = body.text?.trim();
  if (!guestId || !text) {
    return NextResponse.json({ error: 'Guest ID and request text are required.' }, { status: 400 });
  }

  const script = [
    'import json, sys',
    'from dataclasses import asdict',
    'from agent import ConciergeAgent',
    'payload = json.loads(sys.stdin.read())',
    'agent = ConciergeAgent()',
    'response = agent.handle_request(payload["guestId"], payload["text"])',
    'print(json.dumps(asdict(response), ensure_ascii=False))',
  ].join('; ');

  const result = await new Promise<{ stdout: string; stderr: string; code: number | null }>((resolve) => {
    const child = spawn(process.env.PYTHON_EXECUTABLE || 'python', ['-c', script], {
      cwd: process.cwd(),
      windowsHide: true,
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8').on('data', (chunk: string) => { stdout += chunk; });
    child.stderr.setEncoding('utf8').on('data', (chunk: string) => { stderr += chunk; });
    child.stdin.end(JSON.stringify({ guestId, text }));
    child.on('error', (error) => resolve({ stdout, stderr: error.message, code: 1 }));
    child.on('close', (code) => resolve({ stdout, stderr, code }));
  });

  if (result.code !== 0) {
    console.error('Concierge agent failed:', result.stderr);
    return NextResponse.json({ error: 'The concierge agent could not process this request.' }, { status: 500 });
  }

  try {
    return NextResponse.json(JSON.parse(result.stdout.trim()));
  } catch {
    console.error('Concierge agent returned invalid output:', result.stdout, result.stderr);
    return NextResponse.json({ error: 'The concierge agent returned an invalid response.' }, { status: 500 });
  }
}
