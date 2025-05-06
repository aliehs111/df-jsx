import { Disclosure } from '@headlessui/react'
import { ChevronUpIcon, ChevronDownIcon } from '@heroicons/react/20/solid'

export default function QuickStart() {
  return (
    <div className="max-w-3xl mx-auto mb-6">
      <Disclosure as="div" defaultOpen>
        {({ open }) => (
          <div>                             {/* real div instead of <> */}
            <Disclosure.Button
              className="flex w-full justify-between rounded-lg bg-white px-4 py-2 text-left text-lg font-medium text-cyan-800 hover:bg-cyan-50"
            >
              <span>🚀 Getting Started</span>
              {open
                ? <ChevronUpIcon className="h-5 w-5 text-cyan-500" />
                : <ChevronDownIcon className="h-5 w-5 text-cyan-500" />}
            </Disclosure.Button>
            <Disclosure.Panel className="mt-2 space-y-1 rounded-lg bg-white p-4 text-gray-700">
              <ol className="list-decimal list-inside space-y-1">
                <li>Go to <strong>Upload</strong> and pick or drag in a CSV file.</li>
                <li>Preview your data with <code>head</code>, <code>info</code>, or quick plots.</li>
                <li>When you’re happy, hit <strong>Save</strong> so it appears under <strong>My Datasets</strong>.</li>
                <li>Reopen any saved dataset to rerun previews or export results.</li>
              </ol>
            </Disclosure.Panel>
          </div>
        )}
      </Disclosure>
    </div>
  )
}

